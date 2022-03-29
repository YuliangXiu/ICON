import pyvista as pv
import trimesh
import numpy as np
import torch
from body_model import SMPLModel, TetraSMPLModel

# tetgen: https://tetgen.pyvista.org/
# mesh_intersection: https://github.com/vchoutas/torch-mesh-isect.https://github.com/vchoutas/torch-mesh-isect.
import tetgen
from mesh_intersection.bvh_search_tree import BVH

device = torch.device("cuda:0")

# to get adult and kids models
# cd ./ICON
# ln -s ./data/smpl_related/models/smpl ./scripts/tetrahedronize_scripts/smpl

for sex in ['male', 'female', 'neutral']:
    for age in ['kid', 'adult']:
        smpl = SMPLModel(f'./smpl/SMPL_{sex.upper()}.pkl', age)
        trans = np.zeros(smpl.trans_shape)
        beta = np.zeros(smpl.beta_shape)
        pose = np.zeros(smpl.pose_shape)
        pose[1, 2] = 0.5
        pose[2, 2] = -0.5
        smpl.set_params(beta=beta, pose=pose, trans=trans)
        smpl.save_to_obj(f'./data/objs/smpl_{sex}_{age}.obj')
        
        input_mesh = trimesh.Trimesh(smpl.verts, smpl.faces, 
                                    maintain_order=True, process=False)
        vertices = torch.tensor(input_mesh.vertices,
                                dtype=torch.float32, device=device)
        faces = torch.tensor(input_mesh.faces.astype(np.int64),
                            dtype=torch.long,
                            device=device)

        batch_size = 1
        triangles = vertices[faces].unsqueeze(dim=0)
        
        m = BVH(max_collisions=8)

        torch.cuda.synchronize()
        outputs = m(triangles)
        torch.cuda.synchronize()

        outputs = outputs.detach().cpu().numpy().squeeze()

        collisions = outputs[outputs[:, 0] >= 0, :]
        print(collisions.shape)

        isect_vid = input_mesh.faces[collisions.flatten(),:].flatten()
        input_mesh.vertices[isect_vid,:] -= input_mesh.vertex_normals[isect_vid,:] * 1e-3
        input_mesh.export(f'./data/objs/smpl_{sex}_{age}_fix.obj')

        if True:
            fname = f'./data/objs/smpl_{sex}_{age}_fix.obj'
            mesh = pv.read(fname)
            tet = tetgen.TetGen(mesh)
            out = tet.tetrahedralize(order=1)

            np.save(f'./data/tetgen_{sex}_{age}_vertices.npy', out[0])
            np.save(f'./data/tetgen_{sex}_{age}_structure.npy', out[1])
            
            print(out[0].shape, out[1].shape)
            
            tet_smpl_vertices = np.load(f'./data/tetgen_{sex}_{age}_vertices.npy')
            tet_smpl_vertices_added = tet_smpl_vertices[6890:]
            tet_smpl_vertices_orig = tet_smpl_vertices[:6890]
            
            added_weights = []
            added_shape_dirs = []
            added_pose_dirs = []
            
            for v_added in tet_smpl_vertices_added:
                v_added = np.expand_dims(v_added, axis=0)
                dist_list = np.linalg.norm(v_added - tet_smpl_vertices_orig, axis=1)
                min_dist = np.min(dist_list)
                neighbors = dist_list < 2.0 * min_dist
                neighbor_weights = np.exp(-dist_list*dist_list/(2*min_dist*min_dist)) * np.float32(neighbors)
                neighbor_weights_sum = np.sum(neighbor_weights)
                neighbor_weights /= neighbor_weights_sum
                added_weights.append(np.sum(smpl.weights * neighbor_weights[:, np.newaxis], axis=0))
                added_shape_dirs.append(np.sum(smpl.shapedirs * neighbor_weights[:, np.newaxis, np.newaxis], axis=0))
                added_pose_dirs.append(np.sum(smpl.posedirs * neighbor_weights[:, np.newaxis, np.newaxis], axis=0))

            added_weights = np.asarray(added_weights)
            added_shape_dirs = np.asarray(added_shape_dirs)
            added_pose_dirs = np.asarray(added_pose_dirs)
            
            T = np.tensordot(added_weights, smpl.G, axes=[[1], [0]])
            posed_shape_added = tet_smpl_vertices_added - smpl.trans.reshape([1, 3])
            posed_shape_added_h = np.hstack((posed_shape_added, np.ones([posed_shape_added.shape[0], 1])))
            for ti in range(len(T)):
                T[ti] = np.linalg.inv(T[ti])
            rest_shape_added = np.matmul(T, posed_shape_added_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
            
            tetra_smpl_structure = np.load(f'./data/tetgen_{sex}_{age}_structure.npy') 
            np.savez_compressed(f'./data/tetra_{sex}_{age}_smpl.npz', 
                                v_template_added=rest_shape_added, 
                                weights_added=added_weights, 
                                shapedirs_added=added_shape_dirs, 
                                posedirs_added=added_pose_dirs, 
                                tetrahedrons=tetra_smpl_structure)

            np.savetxt(f'./data/tetrahedrons_{sex}_{age}.txt', np.int32(tetra_smpl_structure+1), fmt='%d')

            smpl = TetraSMPLModel(f'./smpl/SMPL_{sex.upper()}.pkl', 
                                f'./data/tetra_{sex}_{age}_smpl.npz',age)
            
            smpl.set_params(beta=beta, pose=pose, trans=trans)
            smpl.save_tetrahedron_to_obj(f'./data/objs/tetrahedron_{sex}_{age}.obj')
