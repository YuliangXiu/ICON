# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from lib.common.seg3d_lossless import Seg3dLossless
from lib.dataset.Evaluator import Evaluator
from lib.net import HGPIFuNet
from lib.common.train_util import *
from lib.common.render import Render
from lib.dataset.mesh_util import SMPLX, update_mesh_shape_prior_losses, get_visibility
import warnings
import logging
import torch
import lib.smplx as smplx
import numpy as np
from torch import nn
from skimage.transform import resize
import pytorch_lightning as pl

torch.backends.cudnn.benchmark = True

logging.getLogger("lightning").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")


class ICON(pl.LightningModule):
    def __init__(self, cfg):
        super(ICON, self).__init__()

        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.lr_G = self.cfg.lr_G

        self.use_sdf = cfg.sdf
        self.prior_type = cfg.net.prior_type
        self.mcube_res = cfg.mcube_res
        self.clean_mesh_flag = cfg.clean_mesh

        self.netG = HGPIFuNet(
            self.cfg,
            self.cfg.projection_mode,
            error_term=nn.SmoothL1Loss() if self.use_sdf else nn.MSELoss(),
        )

        # TODO: replace the renderer from opengl to pytorch3d
        self.evaluator = Evaluator(
            device=torch.device(f"cuda:{self.cfg.gpus[0]}"))

        self.resolutions = (
            np.logspace(
                start=5,
                stop=np.log2(self.mcube_res),
                base=2,
                num=int(np.log2(self.mcube_res) - 4),
                endpoint=True,
            )
            + 1.0
        )
        self.resolutions = self.resolutions.astype(np.int16).tolist()

        self.icon_keys = ["smpl_verts", "smpl_faces", "smpl_vis", "smpl_cmap"]
        self.pamir_keys = ["voxel_verts",
                           "voxel_faces", "pad_v_num", "pad_f_num"]

        self.reconEngine = Seg3dLossless(
            query_func=query_func,
            b_min=[[-1.0, 1.0, -1.0]],
            b_max=[[1.0, -1.0, 1.0]],
            resolutions=self.resolutions,
            align_corners=True,
            balance_value=0.50,
            device=torch.device(f"cuda:{self.cfg.test_gpus[0]}"),
            visualize=False,
            debug=False,
            use_cuda_impl=False,
            faster=True,
        )

        self.render = Render(
            size=512, device=torch.device(f"cuda:{self.cfg.test_gpus[0]}")
        )
        self.smpl_data = SMPLX()

        self.get_smpl_model = lambda smpl_type, gender, age, v_template: smplx.create(
            self.smpl_data.model_dir,
            kid_template_path=osp.join(
                osp.realpath(self.smpl_data.model_dir),
                f"{smpl_type}/{smpl_type}_kid_template.npy",
            ),
            model_type=smpl_type,
            gender=gender,
            age=age,
            v_template=v_template,
            use_face_contour=False,
            ext="pkl",
        )

        self.in_geo = [item[0] for item in cfg.net.in_geo]
        self.in_nml = [item[0] for item in cfg.net.in_nml]
        self.in_geo_dim = [item[1] for item in cfg.net.in_geo]
        self.in_total = self.in_geo + self.in_nml
        self.smpl_dim = cfg.net.smpl_dim

        self.export_dir = None
        self.result_eval = {}

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if "v_num" in tqdm_dict:
            del tqdm_dict["v_num"]
        return tqdm_dict

    # Training related
    def configure_optimizers(self):

        # set optimizer
        weight_decay = self.cfg.weight_decay
        momentum = self.cfg.momentum

        optim_params_G = [
            {"params": self.netG.if_regressor.parameters(), "lr": self.lr_G}
        ]

        if self.cfg.net.use_filter:
            optim_params_G.append(
                {"params": self.netG.F_filter.parameters(), "lr": self.lr_G}
            )

        if self.cfg.net.prior_type == "pamir":
            optim_params_G.append(
                {"params": self.netG.ve.parameters(), "lr": self.lr_G}
            )

        if self.cfg.optim == "Adadelta":

            optimizer_G = torch.optim.Adadelta(
                optim_params_G, lr=self.lr_G, weight_decay=weight_decay
            )

        elif self.cfg.optim == "Adam":

            optimizer_G = torch.optim.Adam(
                optim_params_G, lr=self.lr_G, weight_decay=weight_decay
            )

        elif self.cfg.optim == "RMSprop":

            optimizer_G = torch.optim.RMSprop(
                optim_params_G,
                lr=self.lr_G,
                weight_decay=weight_decay,
                momentum=momentum,
            )

        else:
            raise NotImplementedError

        # set scheduler
        scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_G, milestones=self.cfg.schedule, gamma=self.cfg.gamma
        )

        return [optimizer_G], [scheduler_G]

    def training_step(self, batch, batch_idx):

        if not self.cfg.fast_dev:
            export_cfg(self.logger, self.cfg)

        self.netG.train()

        in_tensor_dict = {
            "sample": batch["samples_geo"].permute(0, 2, 1),
            "calib": batch["calib"],
            "label": batch["labels_geo"].unsqueeze(1),
        }

        for name in self.in_total:
            in_tensor_dict.update({name: batch[name]})

        if self.prior_type == "icon":
            for key in self.icon_keys:
                in_tensor_dict.update({key: batch[key]})
        elif self.prior_type == "pamir":
            for key in self.pamir_keys:
                in_tensor_dict.update({key: batch[key]})
        else:
            pass

        preds_G, error_G = self.netG(in_tensor_dict)

        acc, iou, prec, recall = self.evaluator.calc_acc(
            preds_G.flatten(),
            in_tensor_dict["label"].flatten(),
            0.5,
            use_sdf=self.cfg.sdf,
        )

        # metrics processing
        metrics_log = {
            "train_loss": error_G.item(),
            "train_acc": acc.item(),
            "train_iou": iou.item(),
            "train_prec": prec.item(),
            "train_recall": recall.item(),
        }

        tf_log = tf_log_convert(metrics_log)
        bar_log = bar_log_convert(metrics_log)

        if batch_idx % int(self.cfg.freq_show_train) == 0:

            with torch.no_grad():
                self.render_func(in_tensor_dict, dataset="train")

        metrics_return = {
            k.replace("train_", ""): torch.tensor(v) for k, v in metrics_log.items()
        }

        metrics_return.update(
            {"loss": error_G, "log": tf_log, "progress_bar": bar_log})

        return metrics_return

    def training_epoch_end(self, outputs):

        if [] in outputs:
            outputs = outputs[0]

        # metrics processing
        metrics_log = {
            "train_avgloss": batch_mean(outputs, "loss"),
            "train_avgiou": batch_mean(outputs, "iou"),
            "train_avgprec": batch_mean(outputs, "prec"),
            "train_avgrecall": batch_mean(outputs, "recall"),
            "train_avgacc": batch_mean(outputs, "acc"),
        }

        tf_log = tf_log_convert(metrics_log)

        return {"log": tf_log}

    def validation_step(self, batch, batch_idx):

        self.netG.eval()
        self.netG.training = False

        in_tensor_dict = {
            "sample": batch["samples_geo"].permute(0, 2, 1),
            "calib": batch["calib"],
            "label": batch["labels_geo"].unsqueeze(1),
        }

        for name in self.in_total:
            in_tensor_dict.update({name: batch[name]})

        if self.prior_type == "icon":
            for key in self.icon_keys:
                in_tensor_dict.update({key: batch[key]})
        elif self.prior_type == "pamir":
            for key in self.pamir_keys:
                in_tensor_dict.update({key: batch[key]})
        else:
            pass

        preds_G, error_G = self.netG(in_tensor_dict)

        acc, iou, prec, recall = self.evaluator.calc_acc(
            preds_G.flatten(),
            in_tensor_dict["label"].flatten(),
            0.5,
            use_sdf=self.cfg.sdf,
        )

        if batch_idx % int(self.cfg.freq_show_val) == 0:
            with torch.no_grad():
                self.render_func(in_tensor_dict, dataset="val", idx=batch_idx)

        metrics_return = {
            "val_loss": error_G,
            "val_acc": acc,
            "val_iou": iou,
            "val_prec": prec,
            "val_recall": recall,
        }

        return metrics_return

    def validation_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "val_avgloss": batch_mean(outputs, "val_loss"),
            "val_avgacc": batch_mean(outputs, "val_acc"),
            "val_avgiou": batch_mean(outputs, "val_iou"),
            "val_avgprec": batch_mean(outputs, "val_prec"),
            "val_avgrecall": batch_mean(outputs, "val_recall"),
        }

        tf_log = tf_log_convert(metrics_log)

        return {"log": tf_log}

    def compute_vis_cmap(self, smpl_type, smpl_verts, smpl_faces):

        (xy, z) = torch.as_tensor(smpl_verts).split([2, 1], dim=1)
        smpl_vis = get_visibility(xy, -z, torch.as_tensor(smpl_faces).long())
        if smpl_type == "smpl":
            smplx_ind = self.smpl_data.smpl2smplx(np.arange(smpl_vis.shape[0]))
        else:
            smplx_ind = np.arange(smpl_vis.shape[0])
        smpl_cmap = self.smpl_data.get_smpl_mat(smplx_ind)

        return {
            "smpl_vis": smpl_vis.unsqueeze(0).to(self.device),
            "smpl_cmap": smpl_cmap.unsqueeze(0).to(self.device),
            "smpl_verts": smpl_verts.unsqueeze(0),
        }

    @torch.enable_grad()
    def optim_body(self, in_tensor_dict, batch):

        smpl_model = self.get_smpl_model(
            batch["type"][0], batch["gender"][0], batch["age"][0], None
        ).to(self.device)
        in_tensor_dict["smpl_faces"] = (
            torch.tensor(smpl_model.faces.astype(np.int))
            .long()
            .unsqueeze(0)
            .to(self.device)
        )

        # The optimizer and variables
        optimed_pose = torch.tensor(
            batch["body_pose"][0], device=self.device, requires_grad=True
        )  # [1,23,3,3]
        optimed_trans = torch.tensor(
            batch["transl"][0], device=self.device, requires_grad=True
        )  # [3]
        optimed_betas = torch.tensor(
            batch["betas"][0], device=self.device, requires_grad=True
        )  # [1,10]
        optimed_orient = torch.tensor(
            batch["global_orient"][0], device=self.device, requires_grad=True
        )  # [1,1,3,3]

        optimizer_smpl = torch.optim.SGD(
            [optimed_pose, optimed_trans, optimed_betas, optimed_orient],
            lr=1e-3,
            momentum=0.9,
        )
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl, mode="min", factor=0.5, verbose=0, min_lr=1e-5, patience=5
        )
        loop_smpl = range(50)
        for i in loop_smpl:

            optimizer_smpl.zero_grad()

            # prior_loss, optimed_pose = dataset.vposer_prior(optimed_pose)
            smpl_out = smpl_model(
                betas=optimed_betas,
                body_pose=optimed_pose,
                global_orient=optimed_orient,
                transl=optimed_trans,
                return_verts=True,
            )

            smpl_verts = smpl_out.vertices[0] * 100.0
            smpl_verts = projection(
                smpl_verts, batch["calib"][0], format="tensor")
            smpl_verts[:, 1] *= -1
            # render optimized mesh (normal, T_normal, image [-1,1])
            self.render.load_meshes(
                smpl_verts, in_tensor_dict["smpl_faces"])
            (
                in_tensor_dict["T_normal_F"],
                in_tensor_dict["T_normal_B"],
            ) = self.render.get_rgb_image()

            T_mask_F, T_mask_B = self.render.get_silhouette_image()

            with torch.no_grad():
                (
                    in_tensor_dict["normal_F"],
                    in_tensor_dict["normal_B"],
                ) = self.netG.normal_filter(in_tensor_dict)

            # mask = torch.abs(in_tensor['T_normal_F']).sum(dim=0, keepdims=True) > 0.0
            diff_F_smpl = torch.abs(
                in_tensor_dict["T_normal_F"] - in_tensor_dict["normal_F"]
            )
            diff_B_smpl = torch.abs(
                in_tensor_dict["T_normal_B"] - in_tensor_dict["normal_B"]
            )
            loss = (diff_F_smpl + diff_B_smpl).mean()

            # silhouette loss
            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)[0]
            gt_arr = torch.cat(
                [in_tensor_dict["normal_F"][0], in_tensor_dict["normal_B"][0]], dim=2
            ).permute(1, 2, 0)
            gt_arr = ((gt_arr + 1.0) * 0.5).to(self.device)
            bg_color = (
                torch.Tensor([0.5, 0.5, 0.5]).unsqueeze(
                    0).unsqueeze(0).to(self.device)
            )
            gt_arr = ((gt_arr - bg_color).sum(dim=-1) != 0.0).float()
            loss += torch.abs(smpl_arr - gt_arr).mean()

            # Image.fromarray(((in_tensor_dict['T_normal_F'][0].permute(1,2,0)+1.0)*0.5*255.0).detach().cpu().numpy().astype(np.uint8)).show()

            # loop_smpl.set_description(f"smpl = {loss:.3f}")

            loss.backward(retain_graph=True)
            optimizer_smpl.step()
            scheduler_smpl.step(loss)
            in_tensor_dict["smpl_verts"] = smpl_verts.unsqueeze(0)

        in_tensor_dict.update(
            self.compute_vis_cmap(
                batch["type"][0],
                in_tensor_dict["smpl_verts"][0],
                in_tensor_dict["smpl_faces"][0],
            )
        )

        features, inter = self.netG.filter(in_tensor_dict, return_inter=True)

        return features, inter, in_tensor_dict

    @torch.enable_grad()
    def optim_cloth(self, verts_pr, faces_pr, inter):

        # convert from GT to SDF
        verts_pr -= (self.resolutions[-1] - 1) / 2.0
        verts_pr /= (self.resolutions[-1] - 1) / 2.0

        losses = {
            "cloth": {"weight": 5.0, "value": 0.0},
            "edge": {"weight": 100.0, "value": 0.0},
            "normal": {"weight": 0.2, "value": 0.0},
            "laplacian": {"weight": 100.0, "value": 0.0},
            "smpl": {"weight": 1.0, "value": 0.0},
            "deform": {"weight": 20.0, "value": 0.0},
        }

        deform_verts = torch.full(
            verts_pr.shape, 0.0, device=self.device, requires_grad=True
        )
        optimizer_cloth = torch.optim.SGD(
            [deform_verts], lr=1e-1, momentum=0.9)
        scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_cloth, mode="min", factor=0.1, verbose=0, min_lr=1e-3, patience=5
        )
        # cloth optimization
        loop_cloth = range(100)

        for i in loop_cloth:

            optimizer_cloth.zero_grad()

            self.render.load_meshes(
                verts_pr.unsqueeze(0).to(self.device),
                faces_pr.unsqueeze(0).to(self.device).long(),
                deform_verts,
            )
            P_normal_F, P_normal_B = self.render.get_rgb_image()

            update_mesh_shape_prior_losses(self.render.mesh, losses)
            diff_F_cloth = torch.abs(P_normal_F[0] - inter[:3])
            diff_B_cloth = torch.abs(P_normal_B[0] - inter[3:])
            losses["cloth"]["value"] = (diff_F_cloth + diff_B_cloth).mean()
            losses["deform"]["value"] = torch.topk(
                torch.abs(deform_verts.flatten()), 30
            )[0].mean()

            # Weighted sum of the losses
            cloth_loss = torch.tensor(0.0, device=self.device)
            pbar_desc = ""

            for k in losses.keys():
                if k != "smpl":
                    cloth_loss_per_cls = losses[k]["value"] * \
                        losses[k]["weight"]
                    pbar_desc += f"{k}: {cloth_loss_per_cls:.3f} | "
                    cloth_loss += cloth_loss_per_cls

            # loop_cloth.set_description(pbar_desc)
            cloth_loss.backward(retain_graph=True)
            optimizer_cloth.step()
            scheduler_cloth.step(cloth_loss)

        # convert from GT to SDF
        deform_verts = deform_verts.flatten().detach()
        deform_verts[torch.topk(torch.abs(deform_verts), 30)[
            1]] = deform_verts.mean()
        deform_verts = deform_verts.view(-1, 3).cpu()

        verts_pr += deform_verts
        verts_pr *= (self.resolutions[-1] - 1) / 2.0
        verts_pr += (self.resolutions[-1] - 1) / 2.0

        return verts_pr

    def test_step(self, batch, batch_idx):

        # dict_keys(['dataset', 'subject', 'rotation', 'scale', 'calib',
        #            'normal_F', 'normal_B', 'image', 'T_normal_F', 'T_normal_B',
        #            'z-trans', 'verts', 'faces', 'samples_geo', 'labels_geo',
        #            'smpl_verts', 'smpl_faces', 'smpl_vis', 'smpl_cmap', 'pts_signs',
        #            'type', 'gender', 'age', 'body_pose', 'global_orient', 'betas', 'transl'])

        if self.evaluator._normal_render is None:
            self.evaluator.init_gl()

        self.netG.eval()
        self.netG.training = False
        in_tensor_dict = {}

        # export paths
        mesh_name = batch["subject"][0]
        mesh_rot = batch["rotation"][0].item()
        ckpt_dir = self.cfg.name

        for kid, key in enumerate(self.cfg.dataset.noise_type):
            ckpt_dir += f"_{key}_{self.cfg.dataset.noise_scale[kid]}"

        if self.cfg.optim_cloth:
            ckpt_dir += "_optim_cloth"
        if self.cfg.optim_body:
            ckpt_dir += "_optim_body"

        self.export_dir = osp.join(self.cfg.results_path, ckpt_dir, mesh_name)
        os.makedirs(self.export_dir, exist_ok=True)

        for name in self.in_total:
            if name in batch.keys():
                in_tensor_dict.update({name: batch[name]})

        # update the new T_normal_F/B
        in_tensor_dict.update(
            self.evaluator.render_normal(
                batch["smpl_verts"], batch["smpl_faces"])
        )

        # update the new smpl_vis
        (xy, z) = batch["smpl_verts"][0].split([2, 1], dim=1)
        smpl_vis = get_visibility(
            xy,
            z,
            torch.as_tensor(self.smpl_data.faces).type_as(
                batch["smpl_verts"]).long(),
        )
        in_tensor_dict.update({"smpl_vis": smpl_vis.unsqueeze(0)})

        if self.prior_type == "icon":
            for key in self.icon_keys:
                in_tensor_dict.update({key: batch[key]})
        elif self.prior_type == "pamir":
            for key in self.pamir_keys:
                in_tensor_dict.update({key: batch[key]})
        else:
            pass

        with torch.no_grad():
            if self.cfg.optim_body:
                features, inter, in_tensor_dict = self.optim_body(
                    in_tensor_dict, batch)
            else:
                features, inter = self.netG.filter(
                    in_tensor_dict, return_inter=True)
            sdf = self.reconEngine(
                opt=self.cfg, netG=self.netG, features=features, proj_matrix=None
            )

        # save inter results
        image = (
            in_tensor_dict["image"][0].permute(
                1, 2, 0).detach().cpu().numpy() + 1.0
        ) * 0.5
        smpl_F = (
            in_tensor_dict["T_normal_F"][0].permute(
                1, 2, 0).detach().cpu().numpy()
            + 1.0
        ) * 0.5
        smpl_B = (
            in_tensor_dict["T_normal_B"][0].permute(
                1, 2, 0).detach().cpu().numpy()
            + 1.0
        ) * 0.5
        image_inter = np.concatenate(
            self.tensor2image(512, inter[0]) + [smpl_F, smpl_B, image], axis=1
        )
        Image.fromarray((image_inter * 255.0).astype(np.uint8)).save(
            osp.join(self.export_dir, f"{mesh_rot}_inter.png")
        )

        verts_pr, faces_pr = self.reconEngine.export_mesh(sdf)

        if self.clean_mesh_flag:
            verts_pr, faces_pr = clean_mesh(verts_pr, faces_pr)

        if self.cfg.optim_cloth:
            verts_pr = self.optim_cloth(verts_pr, faces_pr, inter[0].detach())

        verts_gt = batch["verts"][0]
        faces_gt = batch["faces"][0]

        self.result_eval.update(
            {
                "verts_gt": verts_gt,
                "faces_gt": faces_gt,
                "verts_pr": verts_pr,
                "faces_pr": faces_pr,
                "recon_size": (self.resolutions[-1] - 1.0),
                "calib": batch["calib"][0],
            }
        )

        self.evaluator.set_mesh(self.result_eval, scale_factor=1.0)
        self.evaluator.space_transfer()

        chamfer, p2s = self.evaluator.calculate_chamfer_p2s(
            sampled_points=1000)
        normal_consist = self.evaluator.calculate_normal_consist(
            save_demo_img=osp.join(self.export_dir, f"{mesh_rot}_nc.png")
        )

        test_log = {"chamfer": chamfer, "p2s": p2s, "NC": normal_consist}

        return test_log

    def test_epoch_end(self, outputs):

        # make_test_gif("/".join(self.export_dir.split("/")[:-2]))

        accu_outputs = accumulate(
            outputs,
            rot_num=3,
            split={
                "thuman2": (0, 5),
            },
        )

        print(colored(self.cfg.name, "green"))
        print(colored(self.cfg.dataset.noise_scale, "green"))

        self.logger.experiment.add_hparams(
            hparam_dict={"lr_G": self.lr_G, "bsize": self.batch_size},
            metric_dict=accu_outputs,
        )

        np.save(
            osp.join(self.export_dir, "../test_results.npy"),
            accu_outputs,
            allow_pickle=True,
        )

        return accu_outputs

    def tensor2image(self, height, inter):

        all = []
        for dim in self.in_geo_dim:
            img = resize(
                np.tile(
                    ((inter[:dim].cpu().numpy() + 1.0) /
                     2.0).transpose(1, 2, 0),
                    (1, 1, int(3 / dim)),
                ),
                (height, height),
                anti_aliasing=True,
            )

            all.append(img)
            inter = inter[dim:]

        return all

    def render_func(self, in_tensor_dict, dataset="title", idx=0):

        for name in in_tensor_dict.keys():
            in_tensor_dict[name] = in_tensor_dict[name][0:1]

        self.netG.eval()
        features, inter = self.netG.filter(in_tensor_dict, return_inter=True)
        sdf = self.reconEngine(
            opt=self.cfg, netG=self.netG, features=features, proj_matrix=None
        )

        if sdf is not None:
            render = self.reconEngine.display(sdf)

            image_pred = np.flip(render[:, :, ::-1], axis=0)
            height = image_pred.shape[0]

            image_gt = resize(
                ((in_tensor_dict["image"].cpu().numpy()[0] + 1.0) / 2.0).transpose(
                    1, 2, 0
                ),
                (height, height),
                anti_aliasing=True,
            )
            image_inter = self.tensor2image(height, inter[0])
            image = np.concatenate(
                [image_pred, image_gt] + image_inter, axis=1)

            step_id = self.global_step if dataset == "train" else self.global_step + idx
            self.logger.experiment.add_image(
                tag=f"Occupancy-{dataset}/{step_id}",
                img_tensor=image.transpose(2, 0, 1),
                global_step=step_id,
            )

    def test_single(self, batch):

        self.netG.eval()
        self.netG.training = False
        in_tensor_dict = {}

        for name in self.in_total:
            if name in batch.keys():
                in_tensor_dict.update({name: batch[name]})

        if self.prior_type == "icon":
            for key in self.icon_keys:
                in_tensor_dict.update({key: batch[key]})
        elif self.prior_type == "pamir":
            for key in self.pamir_keys:
                in_tensor_dict.update({key: batch[key]})
        else:
            pass

        features, inter = self.netG.filter(in_tensor_dict, return_inter=True)
        sdf = self.reconEngine(
            opt=self.cfg, netG=self.netG, features=features, proj_matrix=None
        )

        verts_pr, faces_pr = self.reconEngine.export_mesh(sdf)

        if self.clean_mesh_flag:
            verts_pr, faces_pr = clean_mesh(verts_pr, faces_pr)

        verts_pr -= (self.resolutions[-1] - 1) / 2.0
        verts_pr /= (self.resolutions[-1] - 1) / 2.0

        return verts_pr, faces_pr, inter
