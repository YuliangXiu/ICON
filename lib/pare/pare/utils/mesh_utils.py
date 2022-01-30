# borrowed from https://github.com/CZ-Wu/GPNet/blob/d8df3b1489800626d4f503f62d2ae8daffe8b686/tools/visualization.py

import trimesh
import numpy as np


def get_world_mesh_list(planeWidth=4,
                        axisHeight=0.7,
                        axisRadius=0.02,
                        add_plane=True):
    groundColor = [220, 220, 220, 255]  # face_colors: [R, G, B, transparency]
    xColor = [255, 0, 0, 128]
    yColor = [0, 255, 0, 128]
    zColor = [0, 0, 255, 128]

    if add_plane:
        ground = trimesh.primitives.Box(
            center=[0, 0, -0.0001], extents=[planeWidth, planeWidth, 0.0002])
        ground.visual.face_colors = groundColor

    xAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    xAxis.apply_transform(matrix=np.mat(((0, 0, 1, axisHeight / 2),
                                         (0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 0,
                                                                       1))))
    xAxis.visual.face_colors = xColor
    yAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    yAxis.apply_transform(matrix=np.mat(((1, 0, 0, 0), (0, 0, -1,
                                                        axisHeight / 2),
                                         (0, 1, 0, 0), (0, 0, 0, 1))))
    yAxis.visual.face_colors = yColor
    zAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    zAxis.apply_transform(matrix=np.mat(((1, 0, 0, 0), (0, 1, 0, 0),
                                         (0, 0, 1, axisHeight / 2), (0, 0, 0,
                                                                     1))))
    zAxis.visual.face_colors = zColor
    xBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3])
    xBox.apply_translation((axisHeight, 0, 0))
    xBox.visual.face_colors = xColor
    yBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3])
    yBox.apply_translation((0, axisHeight, 0))
    yBox.visual.face_colors = yColor
    zBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3])
    zBox.apply_translation((0, 0, axisHeight))
    zBox.visual.face_colors = zColor
    if add_plane:
        worldMeshList = [ground, xAxis, yAxis, zAxis, xBox, yBox, zBox]
    else:
        worldMeshList = [xAxis, yAxis, zAxis, xBox, yBox, zBox]
    return worldMeshList


def get_checkerboard_plane(plane_width=4, num_boxes=9, center=True):

    pw = plane_width / num_boxes
    white = [220, 220, 220, 255]
    black = [35, 35, 35, 255]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            ground = trimesh.primitives.Box(center=[0, 0, -0.0001],
                                            extents=[pw, pw, 0.0002])

            if center:
                c = c[0] + (pw / 2) - (plane_width /
                                       2), c[1] + (pw / 2) - (plane_width / 2)
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.apply_translation([c[0], c[1], 0])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.visual.face_colors = black if ((i + j) % 2) == 0 else white
            meshes.append(ground)

    return meshes


def getNewCoordinate(axisHeight=0.05, axisRadius=0.001):
    xColor = [200, 50, 0, 128]
    yColor = [0, 200, 50, 128]
    zColor = [50, 0, 200, 128]

    xAxis2 = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    xAxis2.apply_transform(matrix=np.mat(((0, 0, 1, axisHeight / 2),
                                          (0, 1, 0, 0), (-1, 0, 0,
                                                         0), (0, 0, 0, 1))))
    xAxis2.visual.face_colors = xColor
    yAxis2 = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    yAxis2.apply_transform(matrix=np.mat(((1, 0, 0, 0), (0, 0, -1,
                                                         axisHeight / 2),
                                          (0, 1, 0, 0), (0, 0, 0, 1))))
    yAxis2.visual.face_colors = yColor
    zAxis2 = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    zAxis2.apply_transform(matrix=np.mat(((1, 0, 0, 0), (0, 1, 0, 0),
                                          (0, 0, 1, axisHeight / 2), (0, 0, 0,
                                                                      1))))
    zAxis2.visual.face_colors = zColor
    xBox2 = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3])
    xBox2.apply_translation((axisHeight, 0, 0))
    xBox2.visual.face_colors = xColor
    yBox2 = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3])
    yBox2.apply_translation((0, axisHeight, 0))
    yBox2.visual.face_colors = yColor
    zBox2 = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3])
    zBox2.apply_translation((0, 0, axisHeight))
    zBox2.visual.face_colors = zColor

    return 1


def meshVisualization(mesh):
    worldMeshList = get_world_mesh_list(planeWidth=0.2,
                                        axisHeight=0.05,
                                        axisRadius=0.001)
    mesh.visual.face_colors = [255, 128, 255, 200]
    worldMeshList.append(mesh)
    scene = trimesh.Scene(worldMeshList)
    scene.show()


def meshPairVisualization(mesh1, mesh2):
    worldMeshList = get_world_mesh_list(planeWidth=0.2,
                                        axisHeight=0.05,
                                        axisRadius=0.001)
    mesh1.visual.face_colors = [255, 128, 255, 200]
    mesh2.visual.face_colors = [255, 255, 128, 200]

    worldMeshList.append((mesh1, mesh2))
    scene = trimesh.Scene(worldMeshList)
    scene.show()


if __name__ == '__main__':
    # meshVisualization(trimesh.convex.convex_hull(np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]])))
    scene = trimesh.Scene(get_checkerboard_plane())
    scene.add_geometry(get_world_mesh_list(add_plane=False))
    scene.show()
