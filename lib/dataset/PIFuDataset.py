import os
import sys
from termcolor import colored
import os.path as osp
import numpy as np
from PIL import Image
import random
import trimesh
import torch
import vedo
from ipdb import set_trace
from kaolin.ops.mesh import check_sign
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from lib.dataset.mesh_util import SMPLX, projection, cal_sdf_batch, get_visibility
from lib.common.render import Render
from lib.dataset.TestDataset import TestDataset
from lib.dataset.hoppeMesh import HoppeMesh
from lib.renderer.mesh import load_fit_body

class PIFuDataset():
    def __init__(self, cfg, split='train', vis=False):

        self.split = split
        self.root = cfg.root
        self.bsize = cfg.batch_size
        self.overfit = cfg.overfit
        
        # for debug, only used in visualize_sampling3D
        self.vis = vis

        self.opt = cfg.dataset
        self.datasets = self.opt.types
        self.input_size = self.opt.input_size
        self.scales = self.opt.scales
        self.workers = cfg.num_threads
        self.prior_type = cfg.net.prior_type
        
        self.noise_type = self.opt.noise_type
        self.noise_scale = self.opt.noise_scale
        
        noise_joints = [4, 5, 7, 8, 13, 14, 16, 17, 18, 19, 20, 21]
        
        self.noise_smpl_idx = []
        self.noise_smplx_idx = []
        
        for idx in noise_joints:
            self.noise_smpl_idx.append(idx * 3)
            self.noise_smpl_idx.append(idx * 3 + 1)
            self.noise_smpl_idx.append(idx * 3 + 2)
            
            self.noise_smplx_idx.append((idx-1) * 3)
            self.noise_smplx_idx.append((idx-1) * 3 + 1)
            self.noise_smplx_idx.append((idx-1) * 3 + 2)

        self.use_sdf = cfg.sdf
        self.sdf_clip = cfg.sdf_clip

        # [(feat_name, channel_num),...]
        self.in_geo = [item[0] for item in cfg.net.in_geo]
        self.in_nml = [item[0] for item in cfg.net.in_nml]

        self.in_geo_dim = [item[1] for item in cfg.net.in_geo]
        self.in_nml_dim = [item[1] for item in cfg.net.in_nml]

        self.in_total = self.in_geo + self.in_nml
        self.in_total_dim = self.in_geo_dim + self.in_nml_dim

        if self.split == 'train':
            self.rotations = np.arange(0, 360, 360 / self.opt.rotation_num).astype(np.int32)
        else:
            self.rotations = range(0, 360, 120)

        self.datasets_dict = {}
        
        for dataset_id, dataset in enumerate(self.datasets):

            mesh_dir = None
            smplx_dir = None
            
            dataset_dir = osp.join(self.root, dataset)

            if dataset in ['thuman2']:
                mesh_dir = osp.join(dataset_dir, "scans")
                smplx_dir = osp.join(dataset_dir, "fits")

            self.datasets_dict[dataset] = {
                "subjects": np.loadtxt(osp.join(dataset_dir, "all.txt"), dtype=str),
                "smplx_dir": smplx_dir,
                "mesh_dir": mesh_dir,
                "scale": self.scales[dataset_id]
            }

        self.subject_list = self.get_subject_list(split)
        self.smplx = SMPLX()

        # PIL to tensor
        self.image_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # PIL to tensor
        self.mask_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0, ), (1.0, ))
        ])

        self.device = torch.device(f"cuda:{cfg.gpus[0]}")
        self.render = Render(size=512, device=self.device)
        
    def render_normal(self, verts, faces, deform_verts=None):
    
        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_simple_mesh(verts, faces, deform_verts)
        return self.render.get_clean_image()
        
        
    def get_subject_list(self, split):

        subject_list = []

        for dataset in self.datasets:

            split_txt = osp.join(self.root, dataset, f'{split}.txt')

            if osp.exists(split_txt):
                print(f"load from {split_txt}")
                subject_list += np.loadtxt(split_txt, dtype=str).tolist()
            else:
                full_txt = osp.join(self.root, dataset, 'all.txt')
                print(f"split {full_txt} into train/val/test")
                
                full_lst = np.loadtxt(full_txt, dtype=str)
                full_lst = [dataset+"/"+item for item in full_lst]
                [train_lst, test_lst, val_lst] = np.split(full_lst, [500, 500+5,])
                
                np.savetxt(full_txt.replace("all", "train"), train_lst, fmt="%s")
                np.savetxt(full_txt.replace("all", "test"), test_lst, fmt="%s")
                np.savetxt(full_txt.replace("all", "val"), val_lst, fmt="%s")
                
                print(f"load from {split_txt}")
                subject_list += np.loadtxt(split_txt, dtype=str).tolist()
                
        if self.split != 'test':
            subject_list += subject_list[:self.bsize-len(subject_list )% self.bsize]
            print(colored(f"total: {len(subject_list)}", "yellow"))
            random.shuffle(subject_list)

        # subject_list = ["thuman2/0499"]
        return subject_list

    def __len__(self):
        return len(self.subject_list) * len(self.rotations)

    def __getitem__(self, index):

        # only pick the first data if overfitting
        if self.overfit:
            index = 0

        rid = index % len(self.rotations)
        mid = index // len(self.rotations)
        
        rotation = self.rotations[rid]
        subject = self.subject_list[mid].split("/")[1]
        dataset = self.subject_list[mid].split("/")[0]
        render_folder = "/".join([dataset + f"_{self.opt.rotation_num}views", subject])

        # setup paths
        data_dict = {
            'dataset': dataset,
            'subject': subject,
            'rotation': rotation,
            'scale': self.datasets_dict[dataset]["scale"],
            'mesh_path': osp.join(self.datasets_dict[dataset]["mesh_dir"], f"{subject}/{subject}.obj"),
            'smplx_path': osp.join(self.datasets_dict[dataset]["smplx_dir"], f"{subject}/smplx_param.pkl"),
            'calib_path': osp.join(self.root, render_folder, 'calib', f'{rotation:03d}.txt'),
            'vis_path': osp.join(self.root, render_folder, 'vis', f'{rotation:03d}.pt'),
            'image_path': osp.join(self.root, render_folder, 'render', f'{rotation:03d}.png')
        }

        # load training data
        data_dict.update(self.load_calib(data_dict))

        # image/normal/depth loader
        for name, channel in zip(self.in_total, self.in_total_dim):

            if f'{name}_path' not in data_dict.keys():
                data_dict.update({
                    f'{name}_path': osp.join(self.root, render_folder, name, f'{rotation:03d}.png')
                })

            # tensor update
            data_dict.update({
                name: self.imagepath2tensor(data_dict[f'{name}_path'], channel, inv=False)
            })

        data_dict.update(self.load_mesh(data_dict))
        data_dict.update(self.get_sampling_geo(data_dict, is_valid=self.split == "val", is_sdf=self.use_sdf))
        data_dict.update(self.load_smpl(data_dict, self.vis))
        
        if not self.vis:
            del data_dict['mesh']
            del data_dict['verts']
            del data_dict['faces']

        path_keys = [
            key for key in data_dict.keys() if '_path' in key or '_dir' in key
        ]
        for key in path_keys:
            del data_dict[key]

        return data_dict

    def imagepath2tensor(self, path, channel=3, inv=False):

        rgba = Image.open(path).convert('RGBA')
        mask = rgba.split()[-1]
        image = rgba.convert('RGB')
        image = self.image_to_tensor(image)
        mask = self.mask_to_tensor(mask)
        image = (image * mask)[:channel]

        return (image * (0.5 - inv) * 2.0).float()

    def load_calib(self, data_dict):
        calib_data = np.loadtxt(data_dict['calib_path'], dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        calib_mat = torch.from_numpy(calib_mat).float()
        return {'calib': calib_mat}

    def load_mesh(self, data_dict):
        mesh_path = data_dict['mesh_path']
        scale = data_dict['scale']

        mesh_ori = trimesh.load(mesh_path,
                                skip_materials=True,
                                process=False,
                                maintain_order=True)
        verts = mesh_ori.vertices * scale
        faces = mesh_ori.faces
        
        vert_normals = np.array(mesh_ori.vertex_normals)
        face_normals = np.array(mesh_ori.face_normals)

        mesh = HoppeMesh(verts, faces, vert_normals, face_normals)

        return {
            'mesh': mesh,
            'verts': torch.as_tensor(mesh.verts).float(),
            'faces': torch.as_tensor(mesh.faces).long()
        }

    def add_noise(self,
                  beta_num,
                  smpl_pose,
                  smpl_betas,
                  noise_type,
                  noise_scale,
                  hashcode):

        np.random.seed(hashcode)
                
        if 'beta' in noise_type:
            if beta_num != 11:
                smpl_betas += (np.random.rand(beta_num) -
                                0.5) * 2.0 * noise_scale[noise_type.index("beta")]
            smpl_betas = smpl_betas.astype(np.float32)

        if 'pose' in noise_type:
            smpl_pose[self.noise_smplx_idx] += (
                np.random.rand(len(self.noise_smplx_idx)) -
                0.5) * 2.0 * np.pi * noise_scale[noise_type.index("pose")]
            smpl_pose = smpl_pose.astype(np.float32)

        return torch.as_tensor(smpl_pose[None, ...]), torch.as_tensor(smpl_betas[None, ...])

    def compute_smpl_verts(self, data_dict, noise_type=None, noise_scale=None):

        dataset = data_dict['dataset']
        smplx_dict = {}

        smplx_param = np.load(data_dict['smplx_path'], allow_pickle=True)
        smplx_pose = smplx_param["body_pose"]  # [1,63]
        smplx_betas = smplx_param["betas"]  # [1,10]
        
        smplx_pose, smplx_betas = self.add_noise(
            smplx_betas.shape[1],
            smplx_pose[0],
            smplx_betas[0],
            noise_type,
            noise_scale,
            hashcode=(hash(f"{data_dict['subject']}_{data_dict['rotation']}"))%(10**8))

        smplx_out, _ = load_fit_body(fitted_path=data_dict['smplx_path'],
                                     scale=self.datasets_dict[dataset]['scale'],
                                     smpl_type='smplx',
                                     smpl_gender='neutral',
                                     noise_dict=dict(betas=smplx_betas,body_pose=smplx_pose))

        smplx_dict.update({"type": "smplx",
                          "gender": 'neutral',
                          "body_pose": torch.as_tensor(smplx_pose),
                          "betas": torch.as_tensor(smplx_betas)})

        return smplx_out.vertices, smplx_dict

    def load_smpl(self, data_dict, vis=False):

        smplx_verts, smplx_dict = self.compute_smpl_verts(
            data_dict, self.noise_type,
            self.noise_scale)  # compute using smpl model

        smplx_verts = projection(smplx_verts, data_dict['calib']).float()
        smplx_faces = torch.as_tensor(self.smplx.faces).long()
        smplx_vis = torch.load(data_dict['vis_path']).float()
        smplx_cmap = torch.as_tensor(np.load(self.smplx.cmap_vert_path)).float()
        
        # get smpl_signs
        query_points = projection(data_dict['samples_geo'],
                                  data_dict['calib']).float()
        
        pts_signs = 2.0 * (check_sign(smplx_verts.unsqueeze(0), 
                                      smplx_faces, 
                                      query_points.unsqueeze(0)).float() - 0.5).squeeze(0)

        return_dict = {
            'smpl_verts': smplx_verts,
            'smpl_faces': smplx_faces,
            'smpl_vis': smplx_vis,
            'smpl_cmap': smplx_cmap,
            'pts_signs': pts_signs
        }
        if smplx_dict is not None:
            return_dict.update(smplx_dict)

        if vis:

            (xy, z) = torch.as_tensor(smplx_verts).to(self.device).split([2, 1], dim=1)
            smplx_vis = get_visibility(xy, z, torch.as_tensor(smplx_faces).to(self.device).long())
            
            T_normal_F, T_normal_B = self.render_normal(
                (smplx_verts*torch.tensor([1.0, -1.0, 1.0])).to(self.device),
                smplx_faces.to(self.device))
            
            return_dict.update({"T_normal_F": T_normal_F.squeeze(0), 
                                "T_normal_B": T_normal_B.squeeze(0)})
            query_points = projection(data_dict['samples_geo'],
                                      data_dict['calib']).float()
            
            smplx_sdf, smplx_norm, smplx_cmap, smplx_vis = cal_sdf_batch(
                smplx_verts.unsqueeze(0).to(self.device),
                smplx_faces.unsqueeze(0).to(self.device),
                smplx_cmap.unsqueeze(0).to(self.device),
                smplx_vis.unsqueeze(0).to(self.device),
                query_points.unsqueeze(0).contiguous().to(self.device))

            return_dict.update({
                'smpl_feat':
                torch.cat(
                    (smplx_sdf[0].detach().cpu(), 
                     smplx_cmap[0].detach().cpu(),
                     smplx_norm[0].detach().cpu(),
                     smplx_vis[0].detach().cpu()),
                    dim=1)
            })

        return return_dict

    def get_sampling_geo(self, data_dict, is_valid=False, is_sdf=False):

        mesh = data_dict['mesh']
        calib = data_dict['calib']

        # Samples are around the true surface with an offset
        n_samples_surface = 4 * self.opt.num_sample_geo
        vert_ids = np.arange(mesh.verts.shape[0])
        thickness_sample_ratio = np.ones_like(vert_ids).astype(np.float32)

        thickness_sample_ratio /= thickness_sample_ratio.sum()

        samples_surface_ids = np.random.choice(vert_ids,
                                               n_samples_surface,
                                               replace=True,
                                               p=thickness_sample_ratio)

        samples_normal_ids = np.random.choice(vert_ids,
                                              self.opt.num_sample_geo // 2,
                                              replace=False,
                                              p=thickness_sample_ratio)

        surf_samples = mesh.verts[samples_normal_ids, :]
        surf_normals = mesh.vert_normals[samples_normal_ids, :]

        samples_surface = mesh.verts[samples_surface_ids, :]

        # Sampling offsets are random noise with constant scale (15cm - 20cm)
        offset = np.random.normal(scale=self.opt.sigma_geo,
                                  size=(n_samples_surface, 1))
        samples_surface += mesh.vert_normals[samples_surface_ids, :] * offset

        # Uniform samples in [-1, 1]
        calib_inv = np.linalg.inv(calib)
        n_samples_space = self.opt.num_sample_geo // 4
        samples_space_img = 2.0 * np.random.rand(n_samples_space, 3) - 1.0
        samples_space = projection(samples_space_img, calib_inv)

        # z-ray direction samples
        if self.opt.zray_type and not is_valid:
            n_samples_rayz = self.opt.ray_sample_num
            samples_surface_cube = projection(samples_surface, calib)
            samples_surface_cube_repeat = np.repeat(samples_surface_cube,
                                                    n_samples_rayz,
                                                    axis=0)

            thickness_repeat = np.repeat(0.5 *
                                         np.ones_like(samples_surface_ids),
                                         n_samples_rayz,
                                         axis=0)

            noise_repeat = np.random.normal(scale=0.40,
                                            size=(n_samples_surface *
                                                  n_samples_rayz, ))
            samples_surface_cube_repeat[:,
                                        -1] += thickness_repeat * noise_repeat
            samples_surface_rayz = projection(samples_surface_cube_repeat,
                                              calib_inv)

            samples = np.concatenate(
                [samples_surface, samples_space, samples_surface_rayz], 0)
        else:
            samples = np.concatenate([samples_surface, samples_space], 0)

        np.random.shuffle(samples)

        # labels: in->1.0; out->0.0.
        if is_sdf:
            sdfs = mesh.get_sdf(samples)
            inside_samples = samples[sdfs < 0]
            outside_samples = samples[sdfs >= 0]

            inside_sdfs = sdfs[sdfs < 0]
            outside_sdfs = sdfs[sdfs >= 0]
        else:
            inside = mesh.contains(samples)
            inside_samples = samples[inside >= 0.5]
            outside_samples = samples[inside < 0.5]

        nin = inside_samples.shape[0]

        if nin > self.opt.num_sample_geo // 2:
            inside_samples = inside_samples[:self.opt.num_sample_geo // 2]
            outside_samples = outside_samples[:self.opt.num_sample_geo // 2]
            if is_sdf:
                inside_sdfs = inside_sdfs[:self.opt.num_sample_geo // 2]
                outside_sdfs = outside_sdfs[:self.opt.num_sample_geo // 2]
        else:
            outside_samples = outside_samples[:(self.opt.num_sample_geo - nin)]
            if is_sdf:
                outside_sdfs = outside_sdfs[:(self.opt.num_sample_geo - nin)]

        if is_sdf:
            samples = np.concatenate(
                [inside_samples, outside_samples, surf_samples], 0)

            labels = np.concatenate([
                inside_sdfs, outside_sdfs, 0.0 * np.ones(surf_samples.shape[0])
            ])

            normals = np.zeros_like(samples)
            normals[-self.opt.num_sample_geo // 2:, :] = surf_normals

            # convert sdf from [-14, 130] to [0, 1]
            # outside: 0, inside: 1
            # Note: Marching cubes is defined on occupancy space (inside=1.0, outside=0.0)

            labels = -labels.clip(min=-self.sdf_clip, max=self.sdf_clip)
            labels += self.sdf_clip
            labels /= (self.sdf_clip * 2)

        else:
            samples = np.concatenate([inside_samples, outside_samples])
            labels = np.concatenate([
                np.ones(inside_samples.shape[0]),
                np.zeros(outside_samples.shape[0])
            ])

            normals = np.zeros_like(samples)

        samples = torch.from_numpy(samples).float()
        labels = torch.from_numpy(labels).float()
        normals = torch.from_numpy(normals).float()

        return {'samples_geo': samples, 'labels_geo': labels}

    def visualize_sampling3D(self, data_dict, mode='vis'):

        # create plot
        vp = vedo.Plotter(title="", size=(1500, 1500), axes=0, bg='white')
        vis_list = []

        assert mode in ['vis', 'sdf', 'normal', 'cmap', 'occ']

        # sdf-1 cmap-3 norm-3 vis-1
        if mode == 'vis':
            labels = data_dict[f'smpl_feat'][:, [-1]]  # visibility
            colors = np.concatenate([labels, labels, labels], axis=1)
        elif mode == 'occ':
            labels = data_dict[f'labels_geo'][...,None] # occupancy
            colors = np.concatenate([labels, labels, labels], axis=1)
        elif mode == 'sdf':
            labels = data_dict[f'smpl_feat'][:, [0]]  # sdf
            labels -= labels.min()
            labels /= labels.max()
            colors = np.concatenate([labels, labels, labels], axis=1)
        elif mode == 'normal':
            labels = data_dict[f'smpl_feat'][:,-4:-1] # normal
            colors = (labels + 1.0 ) * 0.5
        elif mode == 'cmap':
            labels = data_dict[f'smpl_feat'][:,-7:-4] # colormap
            colors = np.array(labels)

        points = projection(data_dict['samples_geo'], data_dict['calib'])
        verts = projection(data_dict['verts'], data_dict['calib'])
        points[:, 1] *= -1
        verts[:, 1] *= -1

        # create a mesh
        mesh = trimesh.Trimesh(verts, data_dict['faces'], process=True)
        mesh.visual.vertex_colors = [128.0, 128.0, 128.0, 255.0]
        vis_list.append(mesh)

        if 'smpl_verts' in data_dict.keys():
            smplx_verts = data_dict['smpl_verts']
            smplx_faces = data_dict['smpl_faces']
            smplx_verts[:, 1] *= -1
            smplx = trimesh.Trimesh(smplx_verts, smplx_faces[:,[0,2,1]], process=False, maintain_order=True)
            smplx.visual.vertex_colors = ((smplx.vertex_normals + 1.0) * 0.5)
            vis_list.append(smplx)

        # create a picure
        img_pos = [1.0, 0.0, -1.0]
        for img_id, img_key in enumerate(['normal_F', 'image', 'T_normal_B']):
            image_arr = (data_dict[img_key].detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255.0
            image_dim = image_arr.shape[0]
            image = vedo.Picture(image_arr).scale(2.0 / image_dim).pos(-1.0, -1.0, img_pos[img_id])
            vis_list.append(image)

        # create a pointcloud
        pc = vedo.Points(points, r=15, c=np.float32(colors))
        vis_list.append(pc)
        
        vp.show(*vis_list, bg="white", axes=1.0, interactive=True)


if __name__ == '__main__':

    import sys
    import os
    import argparse
    sys.path.append(osp.join(osp.dirname(__file__), "../../"))
    from lib.common.config import get_cfg_defaults

    args = get_cfg_defaults()
    args.merge_from_file("../configs/train/icon-filter.yaml")

    # loading cfg file
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--show',
                        action='store_true',
                        help='vis sampler 3D')
    parser.add_argument('-s',
                        '--speed',
                        action='store_true',
                        help='vis sampler 3D')
    parser.add_argument('-l',
                        '--list',
                        action='store_true',
                        help='vis sampler 3D')
    args_c = parser.parse_args()

    dataset = PIFuDataset(args, split='train', vis=args_c.show)
    print(f"Number of subjects :{len(dataset.subject_list)}")
    data_dict = dataset[0]

    if args_c.list:
        for k in data_dict.keys():
            if not hasattr(data_dict[k], "shape"):
                print(f"{k}: {data_dict[k]}")
            else:
                print(f"{k}: {data_dict[k].shape}")
                
    if args_c.show:
        # for item in dataset:
        item = dataset[0]
        dataset.visualize_sampling3D(item, mode='occ')

    if args_c.speed:
        # original: 2 it/s
        # smpl online compute: 2 it/s
        # normal online compute: 1.5 it/s
        from tqdm import tqdm
        for item in tqdm(dataset):
            # pass
            for k in item.keys():
                if 'voxel' in k:
                    if not hasattr(item[k], "shape"):
                        print(f"{k}: {item[k]}")
                    else:
                        print(f"{k}: {item[k].shape}")
            print("--------------------")
