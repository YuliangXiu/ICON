import numpy as np
import json
import itertools
import trimesh
from matplotlib.path import Path
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

def load_segmentation(path, shape):
    """
    Get a segmentation mask for a given image
    Arguments:
        path: path to the segmentation json file
        shape: shape of the output mask
    Returns:
        Returns a segmentation mask
    """
    with open(path) as json_file:
        dict = json.load(json_file)
        segmentations = []
        for key, val in dict.items():
            if not key.startswith('item'):
                continue
            
            # Each item can have multiple polygons. Combine them to one
            # segmentation_coord = list(itertools.chain.from_iterable(val['segmentation']))
            # segmentation_coord = np.round(np.array(segmentation_coord)).astype(int)

            coordinates = []
            for segmentation_coord in val['segmentation']:
                # The format before is [x1,y1, x2, y2, ....]
                x = segmentation_coord[::2]
                y = segmentation_coord[1::2]
                xy = np.vstack((x, y)).T
                coordinates.append(xy)
            
            segmentations.append({'type': val['category_name'], 'type_id': val['category_id'], 'coordinates': coordinates})

        return segmentations

def smpl_to_recon_labels(recon, smpl, k=1):
    """
    Get the bodypart labels for the recon object by using the labels from the corresponding smpl object
    Arguments:
        recon: trimesh object (fully clothed model)
        shape: trimesh object (smpl model)
        k: number of nearest neighbours to use
    Returns:
        Returns a dictionary containing the bodypart and the corresponding indices
    """
    smpl_vert_segmentation = json.load(open('../lib/common/smpl_vert_segmentation.json'))
    n = smpl.vertices.shape[0]
    y = np.array([None] * n)
    for key, val in smpl_vert_segmentation.items():
        y[val] = key

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(smpl.vertices, y)

    y_pred = classifier.predict(recon.vertices)

    recon_labels = {}
    for key in smpl_vert_segmentation.keys():
        recon_labels[key] = list(np.argwhere(y_pred == key).flatten().astype(int))
    
    return recon_labels

def extract_cloth(recon, segmentation, K, R, t, smpl = None):
    """
    Extract a portion of a mesh using 2d segmentation coordinates
    Arguments:
        recon: fully clothed mesh
        seg_coord: segmentation coordinates in 2D (NDC)
        K: intrinsic matrix of the projection
        R: rotation matrix of the projection
        t: translation vector of the projection
    Returns:
        Returns a submesh using the segmentation coordinates
    """
    seg_coord = segmentation['coord_normalized']
    mesh = trimesh.Trimesh(recon.vertices, recon.faces)
    extrinsic = np.zeros((3,4))
    extrinsic[:3, :3] = R
    extrinsic[:,3] = t
    P = K[:3, :3] @ extrinsic

    P_inv = np.linalg.pinv(P)

    ### Each segmentation can contain multiple polygons
    ### We need to check them separately
    points_so_far = []
    faces = recon.faces
    for polygon in seg_coord:
        n = len(polygon)
        coords_h = np.hstack((polygon, np.ones((n,1))))
        # Apply the inverse projection on homogeneus 2D coordinates to get the corresponding 3d Coordinates
        XYZ = P_inv @ coords_h[:,:, None]
        XYZ = XYZ.reshape((XYZ.shape[0], XYZ.shape[1]))
        XYZ = XYZ[:, :3] / XYZ[:,3, None]

        p = Path(XYZ[:, :2])

        grid = p.contains_points(recon.vertices[:,:2])
        indeces = np.argwhere(grid == True)
        points_so_far += list(indeces.flatten())

    if smpl is not None:
        num_verts = recon.vertices.shape[0]
        recon_labels = smpl_to_recon_labels(recon, smpl)
        body_parts_to_remove = ['rightHand', 'leftToeBase', 'leftFoot', 'rightFoot', 'head', 'leftHandIndex1', 'rightHandIndex1', 'rightToeBase', 'leftHand', 'rightHand']
        type = segmentation['type_id'] 

        # Remove additional bodyparts that are most likely not part of the segmentation but might intersect (e.g. hand in front of torso)
        # https://github.com/switchablenorms/DeepFashion2
        # Short sleeve clothes
        if type == 1 or type == 3 or type == 10:
            body_parts_to_remove += ['leftForeArm', 'rightForeArm']
        # No sleeves at all or lower body clothes
        elif type == 5 or type == 6 or type == 12 or type == 13 or type == 8 or type == 9:
            body_parts_to_remove += ['leftForeArm', 'rightForeArm', 'leftArm', 'rightArm']
        # Shorts
        elif type == 7:
            body_parts_to_remove += ['leftLeg', 'rightLeg', 'leftForeArm', 'rightForeArm', 'leftArm', 'rightArm']
        

        verts_to_remove = list(itertools.chain.from_iterable([recon_labels[part] for part in body_parts_to_remove]))

        label_mask = np.zeros(num_verts, dtype=bool)
        label_mask[verts_to_remove] = True

        seg_mask = np.zeros(num_verts, dtype=bool)
        seg_mask[points_so_far] = True

        # Remove points that belong to other bodyparts
        # If a vertice in pointsSoFar is included in the bodyparts to remove, then these points should be removed
        extra_verts_to_remove = np.array(list(seg_mask) and list(label_mask))

        combine_mask = np.zeros(num_verts, dtype=bool)
        combine_mask[points_so_far] = True
        combine_mask[extra_verts_to_remove] = False

        all_indices = np.argwhere(combine_mask == True).flatten()
    
    i_x = np.where(np.in1d(faces[:,0], all_indices))[0]
    i_y = np.where(np.in1d(faces[:,1], all_indices))[0]
    i_z = np.where(np.in1d(faces[:,2], all_indices))[0]

    faces_to_keep = np.array(list(set(i_x).union(i_y).union(i_z)))
    mask = np.zeros(len(recon.faces), dtype=bool)
    if len(faces_to_keep) > 0:
        mask[faces_to_keep] = True

        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()

        mesh.rezero()

        return mesh
    
    return None
