import argparse
import os
import sys
import time
import trimesh
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from lib.renderer.mesh import load_fit_body
from lib.dataset.mesh_util import projection, load_calib, get_visibility

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--subject', type=str, help='subject name')
parser.add_argument(
    '-o', '--out_dir', type=str, help='output dir')
parser.add_argument(    
    '-r', '--rotation', type=str, help='rotation num')
parser.add_argument(    
    '-m', '--mode', type=str, help='gen/debug')

args = parser.parse_args()

subject = args.subject
save_folder = args.out_dir
rotation = int(args.rotation)

dataset = save_folder.split("/")[-1].split("_")[0]

format = 'obj'
scale = 180.0

mesh_file = os.path.join(f'../data/{dataset}/scans/{subject}', f'{subject}.{format}')
fit_file = f'../data/{dataset}/fits/{subject}/smplx_param.pkl'
    
rescale_fitted_body, _ = load_fit_body(fit_file, 
                                        scale, 
                                        smpl_type='smplx', 
                                        smpl_gender='neutral')

smpl_verts = torch.from_numpy(rescale_fitted_body.vertices).cuda().float()
smpl_faces = torch.from_numpy(rescale_fitted_body.faces).cuda().long()

for y in range(0, 360, 360//rotation):
    
    
    calib_file = os.path.join(
        f'{save_folder}/{subject}/calib', f'{y:03d}.txt')
    vis_file = os.path.join(
        f'{save_folder}/{subject}/vis', f'{y:03d}.pt')
    
    os.makedirs(os.path.dirname(vis_file), exist_ok=True)
    
    calib = load_calib(calib_file).cuda()
    calib_verts = projection(smpl_verts, calib, format='tensor')
    (xy,z) = calib_verts.split([2,1],dim=1)
    smpl_vis = get_visibility(xy, z, smpl_faces)
    
    if args.mode == 'debug':
        mesh = trimesh.Trimesh(smpl_verts.cpu().numpy(), smpl_faces.cpu().numpy(), process=False)
        mesh.visual.vertex_colors = torch.tile(smpl_vis, (1,3)).numpy()
        mesh.export(vis_file.replace("pt", "obj"))
    
    torch.save(smpl_vis, vis_file)
        
done_jobs = len(os.listdir(save_folder))
all_jobs = len(os.listdir(f"../data/{dataset}/scans"))
print(
    f"Finish visibility computing {subject}| {done_jobs}/{all_jobs} | Time: {(time.time()-t0):.0f} secs")
