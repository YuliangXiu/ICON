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

from pytorch3d.renderer import (
    BlendParams,
    blending,
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    PointsRasterizationSettings,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizer,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
from lib.dataset.mesh_util import SMPLX, get_visibility

import lib.common.render_utils as util
import torch
import numpy as np
from PIL import Image
from pytorch3d.io import load_obj
import os
import cv2
import math
from termcolor import colored


def image2vid(images, vid_path):

    w, h = images[0].size
    videodims = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(vid_path, fourcc, 30, videodims)
    for image in images:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()


def query_color(verts, faces, image, device):
    """query colors from points and image

    Args:
        verts ([B, 3]): [query verts]
        faces ([M, 3]): [query faces]
        image ([B, 3, H, W]): [full image]

    Returns:
        [np.float]: [return colors]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)

    (xy, z) = verts.split([2, 1], dim=1)
    visibility = get_visibility(xy, z, faces[:, [0, 2, 1]])
    uv = xy.unsqueeze(0).unsqueeze(2)  # [B, N, 2]
    uv = uv * torch.tensor([1.0, -1.0]).type_as(uv)
    colors = torch.nn.functional.grid_sample(
        image, uv, align_corners=True
    )  # [B, C, N, 1]
    colors = (
        (colors[0, :, :, 0].permute(1, 0) + 1.0) * 0.5 * 255.0
    ).detach().cpu() * visibility

    # mesh = trimesh.Trimesh(verts.detach().cpu(), faces.detach().cpu(), process=False, maintains_order=True)
    # mesh.visual.vertex_colors = colors
    # mesh.show()

    return colors


class cleanShader(torch.nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"

            raise ValueError(msg)

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(
            texels, fragments, blend_params, znear=-256, zfar=256
        )

        return images


class Render:
    def __init__(self, size=512, device=torch.device("cuda:0")):
        self.device = device
        self.mesh_y_center = 100.0
        self.dis = 100.0
        self.scale = 1.0
        self.size = size
        self.cam_pos = [(0, 100, 100)]

        self.mesh = None
        self.deform_mesh = None
        self.pcd = None
        self.renderer = None
        self.meshRas = None
        self.type = None
        self.knn = None
        self.knn_inverse = None

        self.smpl_seg = None
        self.smpl_cmap = None

        self.smplx = SMPLX()

        self.uv_rasterizer = util.Pytorch3dRasterizer(self.size)

    def get_camera(self, cam_id):

        R, T = look_at_view_transform(
            eye=[self.cam_pos[cam_id]],
            at=((0, self.mesh_y_center, 0),),
            up=((0, 1, 0),),
        )

        camera = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            znear=100.0,
            zfar=-100.0,
            max_y=100.0,
            min_y=-100.0,
            max_x=100.0,
            min_x=-100.0,
            scale_xyz=(self.scale * np.ones(3),),
        )

        return camera

    def init_renderer(self, camera, type="clean_mesh", bg="gray"):

        if "mesh" in type:

            # rasterizer
            self.raster_settings_mesh = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                faces_per_pixel=30,
            )
            self.meshRas = MeshRasterizer(
                cameras=camera, raster_settings=self.raster_settings_mesh
            )

        if bg == "black":
            blendparam = BlendParams(1e-4, 1e-4, (0.0, 0.0, 0.0))
        elif bg == "white":
            blendparam = BlendParams(1e-4, 1e-8, (1.0, 1.0, 1.0))
        elif bg == "gray":
            blendparam = BlendParams(1e-4, 1e-8, (0.5, 0.5, 0.5))

        if type == "ori_mesh":

            lights = PointLights(
                device=self.device,
                ambient_color=((0.8, 0.8, 0.8),),
                diffuse_color=((0.2, 0.2, 0.2),),
                specular_color=((0.0, 0.0, 0.0),),
                location=[[0.0, 200.0, 0.0]],
            )

            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=camera,
                    lights=lights,
                    blend_params=blendparam,
                ),
            )

        if type == "silhouette":
            self.raster_settings_silhouette = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * 5e-5,
                faces_per_pixel=50,
                cull_backfaces=True,
            )

            self.silhouetteRas = MeshRasterizer(
                cameras=camera, raster_settings=self.raster_settings_silhouette
            )
            self.renderer = MeshRenderer(
                rasterizer=self.silhouetteRas, shader=SoftSilhouetteShader()
            )

        if type == "pointcloud":
            self.raster_settings_pcd = PointsRasterizationSettings(
                image_size=self.size, radius=0.006, points_per_pixel=10
            )

            self.pcdRas = PointsRasterizer(
                cameras=camera, raster_settings=self.raster_settings_pcd
            )
            self.renderer = PointsRenderer(
                rasterizer=self.pcdRas,
                compositor=AlphaCompositor(background_color=(0, 0, 0)),
            )

        if type == "clean_mesh":

            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=cleanShader(
                    device=self.device, cameras=camera, blend_params=blendparam
                ),
            )

    def load_mesh(self, verts, faces, verts_rgb, verts_dense=None, load_seg=False):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3]): verts
            faces ([N,3]): faces
            verts_rgb ([N,3]): verts colors
            verts_dense ([N,3], optinoal): verts dense correspondense results. Defaults to None.
            load_seg (bool, optional): needs to render seg or not. Defaults to False.
        """

        self.type = type
        self.load_seg = load_seg

        # data format convert
        if not torch.is_tensor(verts):
            verts = torch.as_tensor(verts).float().unsqueeze(0).to(self.device)
            faces = torch.as_tensor(faces).int().unsqueeze(0).to(self.device)
            if verts_rgb is not None:
                verts_rgb = (
                    torch.as_tensor(verts_rgb)[:, :3]
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
        else:
            verts = verts.float().unsqueeze(0).to(self.device)
            faces = faces.int().unsqueeze(0).to(self.device)
            if verts_rgb is not None:
                verts_rgb = verts_rgb[:, :3].float(
                ).unsqueeze(0).to(self.device)

        # dense correspondence results data format convert
        if verts_dense is not None:
            if not torch.is_tensor(verts_dense):
                verts_dense = torch.from_numpy(verts_dense)
            verts_dense = verts_dense[:, :3].unsqueeze(0).to(self.device)

        # camera setting
        self.mesh_y_center = (
            0.5 * (verts.max(dim=1)[0][0, 1] + verts.min(dim=1)[0][0, 1])
        ).item()
        self.scale = 90.0 / (self.mesh_y_center + 1e-10)
        self.cam_pos = [
            (0, self.mesh_y_center, self.dis),
            (self.dis, self.mesh_y_center, 0),
            (0, self.mesh_y_center, -self.dis),
            (-self.dis, self.mesh_y_center, 0),
        ]

        # self.verts is for UV rendering, so it is [smpl_num, 3]
        # verts is for normal rendering, so it is [sample_num, 3]

        if verts_rgb is not None:
            self.type = "color"
            textures = TexturesVertex(verts_features=verts_rgb)
            self.verts = verts_rgb.squeeze(0)[self.knn].squeeze(1)

        self.mesh = Meshes(verts=verts, faces=faces,
                           textures=textures).to(self.device)

        _, faces, aux = load_obj(self.smplx.tpose_path, device=self.device)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        self.uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        self.verts = self.verts[None, ...]  # (N, V, 3)
        self.faces = faces.verts_idx[None, ...]  # (N, F, 3)

        # uv coords
        uvcoords = torch.cat(
            [uvcoords, uvcoords[:, :, 0:1] * 0.0 + 1.0], -1
        )  # [bz, ntv, 3]
        self.uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]

    def load_simple_mesh(self, verts, faces, deform_verts=None):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3]): verts
            faces ([N,3]): faces
            offset ([N,3]): offset
        """

        # camera setting
        self.scale = 100.0
        self.mesh_y_center = 0.0

        self.cam_pos = [
            (0, self.mesh_y_center, 100.0),
            (100.0, self.mesh_y_center, 0),
            (0, self.mesh_y_center, -100.0),
            (-100.0, self.mesh_y_center, 0),
        ]

        self.type = "color"

        if not torch.is_tensor(verts):
            verts = torch.tensor(verts)
        if not torch.is_tensor(faces):
            faces = torch.tensor(faces)

        if verts.ndimension() == 2:
            verts = verts.unsqueeze(0).float()
        if faces.ndimension() == 2:
            faces = faces.unsqueeze(0).long()

        verts = verts.to(self.device)
        faces = faces.to(self.device)

        # verts_rgb = (compute_normal_batch(verts, faces) + 1.0) * 0.5

        if deform_verts is not None:

            deform_verts_copy = deform_verts.clone()
            false_ids = torch.topk(torch.abs(deform_verts).sum(dim=1), 30)[1]
            deform_verts_copy[false_ids] = deform_verts_copy.mean(dim=0)

            self.mesh = (
                Meshes(verts, faces).to(
                    self.device).offset_verts(deform_verts_copy)
            )
        else:
            self.mesh = Meshes(verts, faces).to(self.device)

        textures = TexturesVertex(
            verts_features=(self.mesh.verts_normals_padded() + 1.0) * 0.5
        )
        self.mesh.textures = textures

    def load_pcd(self, verts, verts_rgb):
        """load pointcloud into the pytorch3d renderer

        Args:
            verts ([N,3]): verts
            verts_rgb ([N,3]): verts colors
        """

        # data format convert
        if not torch.is_tensor(verts):
            verts = torch.as_tensor(verts).float().unsqueeze(0).to(self.device)
            if verts_rgb is not None:
                verts_rgb = (
                    torch.as_tensor(verts_rgb)[:, :3]
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
        else:
            verts = verts.float().unsqueeze(0).to(self.device)
            if verts_rgb is not None:
                verts_rgb = verts_rgb[:, :3].float(
                ).unsqueeze(0).to(self.device)

        # camera setting
        self.mesh_y_center = (
            0.5 * (verts.max(dim=1)[0][0, 1] + verts.min(dim=1)[0][0, 1])
        ).item()
        self.scale = 90.0 / (self.mesh_y_center + 1e-10)
        self.cam_pos = [
            (0, self.mesh_y_center, self.dis),
            (self.dis, self.mesh_y_center, 0),
            (0, self.mesh_y_center, -self.dis),
            (-self.dis, self.mesh_y_center, 0),
        ]

        pcd = Pointclouds(points=verts, features=verts_rgb).to(self.device)
        return pcd

    def get_image(self):
        images = torch.zeros((self.size, self.size * len(self.cam_pos), 3)).to(
            self.device
        )
        for cam_id in range(len(self.cam_pos)):
            self.init_renderer(self.get_camera(cam_id), "ori_mesh", "gray")
            images[:, self.size * cam_id: self.size * (cam_id + 1), :] = self.renderer(
                self.mesh
            )[0, :, :, :3]

        return images.cpu().numpy()

    def get_clean_image(self, cam_ids=[0, 2]):

        images = []
        for cam_id in range(len(self.cam_pos)):
            if cam_id in cam_ids:
                self.init_renderer(self.get_camera(
                    cam_id), "clean_mesh", "gray")
                if len(cam_ids) == 4:
                    rendered_img = (
                        self.renderer(self.mesh)[
                            0:1, :, :, :3].permute(0, 3, 1, 2)
                        - 0.5
                    ) * 2.0
                else:
                    rendered_img = (
                        self.renderer(self.mesh)[
                            0:1, :, :, :3].permute(0, 3, 1, 2)
                        - 0.5
                    ) * 2.0
                if cam_id == 2 and len(cam_ids) == 2:
                    rendered_img = torch.flip(rendered_img, dims=[3])
                images.append(rendered_img)

        return images

    def get_rendered_video(self, images, save_path):

        self.cam_pos = []
        for angle in range(360):
            self.cam_pos.append(
                (
                    100.0 * math.cos(np.pi / 180 * angle),
                    self.mesh_y_center,
                    100.0 * math.sin(np.pi / 180 * angle),
                )
            )

        old_shape = np.array(images[0].shape[:2])
        new_shape = np.around(
            (self.size / old_shape[0]) * old_shape).astype(np.int)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            save_path, fourcc, 30, (self.size +
                                    new_shape[1] * len(images), self.size)
        )

        print(
            colored(
                f"exporting video {os.path.basename(save_path)}, please wait for a while...",
                "blue",
            )
        )

        for cam_id in range(len(self.cam_pos)):
            self.init_renderer(self.get_camera(cam_id), "clean_mesh", "gray")
            rendered_img = (
                (self.renderer(self.mesh)[0, :, :, :3] * 255.0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            img_lst = [
                np.array(Image.fromarray(img).resize(new_shape[::-1])).astype(np.uint8)[
                    :, :, [2, 1, 0]
                ]
                for img in images
            ]
            img_lst.append(rendered_img)
            final_img = np.concatenate(img_lst, axis=1)
            video.write(final_img)

        video.release()

    def get_silhouette_image(self, cam_ids=[0, 2]):

        images = []
        for cam_id in range(len(self.cam_pos)):
            if cam_id in cam_ids:
                self.init_renderer(self.get_camera(cam_id), "silhouette")
                rendered_img = self.renderer(self.mesh)[0:1, :, :, 3]
                if cam_id == 2 and len(cam_ids) == 2:
                    rendered_img = torch.flip(rendered_img, dims=[2])
                images.append(rendered_img)

        return images

    def get_image_pcd(self, pcd):
        images = torch.zeros((self.size, self.size * len(self.cam_pos), 3)).to(
            self.device
        )
        for cam_id in range(len(self.cam_pos)):
            self.init_renderer(self.get_camera(cam_id), "pointcloud")
            images[:, self.size * cam_id: self.size * (cam_id + 1), :] = self.renderer(
                pcd
            )[0, :, :, :3]

        return images.cpu().numpy()

    def get_texture(self, smpl_color=None):
        """
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        """

        if self.type == "color":
            assert smpl_color is not None, "smpl_color argument should not be empty"

        batch_size = self.verts.shape[0]
        face_vertices = util.face_vertices(
            self.verts, self.faces.expand(batch_size, -1, -1)
        )
        uv_vertices = self.uv_rasterizer(
            self.uvcoords.expand(batch_size, -1, -1),
            self.uvfaces.expand(batch_size, -1, -1),
            face_vertices,
        )[:, :3]
        uv_vertices = uv_vertices.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if self.type == "dense":
            face_vertices = util.face_vertices(
                self.smpl_cmap[None, ...], self.faces.expand(
                    batch_size, -1, -1)
            )
        elif self.type == "color":
            face_vertices = util.face_vertices(
                smpl_color[:, :, :3].to(self.device),
                self.faces.expand(batch_size, -1, -1),
            )
        else:
            face_vertices = util.face_vertices(
                self.smpl_seg[None, ...], self.faces.expand(batch_size, -1, -1)
            )

        uv_vertices_cmap = self.uv_rasterizer(
            self.uvcoords.expand(batch_size, -1, -1),
            self.uvfaces.expand(batch_size, -1, -1),
            face_vertices,
        )[:, :3]

        uv_vertices_cmap = uv_vertices_cmap.squeeze(
            0).permute(1, 2, 0).cpu().numpy()

        return np.concatenate(
            (np.flip(uv_vertices, 0), np.flip(uv_vertices_cmap, 0)), axis=1
        )
