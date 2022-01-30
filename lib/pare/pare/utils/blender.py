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

import os
import re
import bpy
import sys
import argparse
import numpy as np
from math import radians
from mathutils import Matrix, Quaternion


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """
    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1:]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def get_colors(c):
    colors = {
        'pink': np.array([197, 27, 125]),
        'light_pink': np.array([233, 163, 201]),
        'light_green': np.array([161, 215, 106]),
        'green': np.array([77, 146, 33]),
        'red': np.array([215, 48, 39]),
        'light_red': np.array([252, 146, 114]),
        'light_orange': np.array([252, 141, 89]),
        'purple': np.array([118, 42, 131]),
        'light_purple': np.array([175, 141, 195]),
        'light_blue': np.array([145, 191, 219]),
        'blue': np.array([69, 117, 180]),
        'gray': np.array([130, 130, 130]),
        'white': np.array([255, 255, 255]),
        'turkuaz': np.array([50, 134, 204]),
    }
    return colors[c]


##################################################
# Helper functions
##################################################


# Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
#   Source: smpl/plugins/blender/corrective_bpy_sh.py
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)


##################################################
# Set camera extrinsics from OpenCV camera parameters
#
# NOTE: THIS CODE DOES NOT WORK PROPERLY WHEN BOTH rt and t !=0
#       It is better to apply translation+rotation to objects and keep camera at default looking along Y
#
##################################################
def set_camera_extrinsics_from_opencv(camera, rt, t):
    # Map OpenCV translation to Blender coordinates
    x = t[0]
    y = t[2]
    z = -t[1]

    # OpenCV applies translation to all vertices
    # In Blender we need to move camera in opposite direction to achieve same effect
    camera.location = (-x, -y, -z)

    mat = Rodrigues(np.array(rt))

    # Apply rotation first to Blender camera system (Looking down negative world Z-Axis)
    quat = Matrix(mat).to_quaternion()
    quat = Quaternion((-quat.w, quat.x, -quat.y, -quat.z))

    camera.rotation_mode = 'QUATERNION'

    # Map rotation in Blender camera system to world rotation
    # Default camera looking along positive Y axis with XY being the ground plane

    quat_world_from_camera = Quaternion((1.0, 0.0, 0.0), radians(90))
    camera.rotation_quaternion = quat_world_from_camera @ quat


##################################################
# Set camera intrinsics from OpenCV camera parameters
##################################################
def set_camera_intrinsics_from_opencv(camera, width, height, fx, fy, cx, cy):
    # Resolution
    bpy.data.scenes['Scene'].render.resolution_x = width
    bpy.data.scenes['Scene'].render.resolution_y = height

    # Use horizontal sensor fit
    camera.data.sensor_fit = 'HORIZONTAL'

    # Focal length
    camera.data.lens_unit = 'MILLIMETERS'
    sensor_width_in_mm = camera.data.sensor_width
    camera.data.lens = (fx / width) * sensor_width_in_mm

    # TODO: fx != fy

    # Camera principal point
    #
    # Note: For nonrectangular images a shift_y of 0.5 is not moving the resulting image by half the height
    #       shift_y=1 is the same pixel shift as shift_x=1
    #       (Blender 2.79b)
    camera.data.shift_x = -(cx / width - 0.5)
    camera.data.shift_y = (cy - 0.5 * height) / width

    return


def np_array_from_image(img_name):
    img = bpy.data.images.load(
        img_name, check_existing=True)  # bpy.data.images[img_name]
    img = np.array(img.pixels[:])
    return img


def save_image(fname, img):
    output_image = bpy.data.images.new('save_img',
                                       height=img.shape[0],
                                       width=img.shape[1])
    output_image.file_format = 'PNG'
    output_image.pixels = img.ravel()
    output_image.filepath_raw = fname
    output_image.save()


def overlay_smooth(img, render):
    img = np_array_from_image(img)
    render = np_array_from_image(render)
    img_size = int(np.sqrt(render.shape[0] // 4))

    render = render.reshape((img_size, img_size, 4))
    img = img.reshape((img_size, img_size, 4))

    # breakpoint()

    m = render[:, :, -1:]  #  / 255.
    i = img[:, :, :3] * (1 - m) + render[:, :, :3] * m
    i = np.clip(i, 0., 1.)  # .astype(np.uint8)
    i = np.concatenate([i, np.zeros((img_size, img_size, 1))], axis=-1)
    return i


def process_data(object_path,
                 output_dir,
                 numpy_path,
                 wireframe,
                 quads,
                 width,
                 height,
                 rotation_angle=0):
    ####################
    # Object
    ####################
    print("Loading obj: " + object_path)
    bpy.ops.import_scene.obj(filepath=object_path)
    object = bpy.context.selected_objects[0]
    object.data.materials[0] = bpy.data.materials['Body']

    object.rotation_euler[2] = radians(rotation_angle)

    if quads:
        bpy.context.view_layer.objects.active = object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.shade_smooth()

    # Mark freestyle edges
    bpy.context.view_layer.objects.active = object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    ####################
    # Camera
    ####################
    # numpy_path = object_path.replace('.obj', '.npy')
    print("Loading npy: " + numpy_path)

    t = np.load(numpy_path)

    # NOTE: Negate first value for proper background overlay
    camera_translation = (-t[0], t[1], t[2])

    camera = bpy.data.objects['Camera']

    set_camera_extrinsics_from_opencv(camera, (0.0, 0.0, 0.0),
                                      camera_translation)

    scale = 1

    set_camera_intrinsics_from_opencv(camera, width, height, 5000 * scale,
                                      5000 * scale, width / 2, height / 2)

    ####################
    # Render
    ####################
    output_file = os.path.basename(object_path).replace(".obj", "_render.png")
    bpy.context.scene.render.filepath = os.path.join(output_dir, output_file)
    # bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.render.render(write_still=True)

    if object_path.endswith('_rot.obj'):
        pass
    else:
        # Overlay image
        filepath = os.path.join(output_dir, output_file)
        overlay_img = overlay_smooth(filepath.replace('_render.png', '.jpg'),
                                     filepath)
        save_image(filepath.replace('_render.png', '_overlay.png'),
                   overlay_img)

    # Delete last selected object from scene
    object.select_set(True)
    bpy.ops.object.delete()


def render_turntable(object_path,
                     output_fname,
                     numpy_path,
                     wireframe,
                     quads,
                     width,
                     height,
                     rotation_angle=0):
    ####################
    # Object
    ####################
    print("Loading obj: " + object_path)
    bpy.ops.import_scene.obj(filepath=object_path)
    object = bpy.context.selected_objects[0]
    object.data.materials[0] = bpy.data.materials['Body']

    object.rotation_euler[2] = radians(rotation_angle)

    if quads:
        bpy.context.view_layer.objects.active = object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.shade_smooth()

    # Mark freestyle edges
    bpy.context.view_layer.objects.active = object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    ####################
    # Camera
    ####################
    # numpy_path = object_path.replace('.obj', '.npy')
    print("Loading npy: " + numpy_path)

    t = np.load(numpy_path)

    # NOTE: Negate first value for proper background overlay
    camera_translation = (-t[0], t[1], t[2])

    camera = bpy.data.objects['Camera']

    set_camera_extrinsics_from_opencv(camera, (0.0, 0.0, 0.0),
                                      camera_translation)

    scale = 1

    set_camera_intrinsics_from_opencv(camera, width, height, 5000 * scale,
                                      5000 * scale, width / 2, height / 2)

    ####################
    # Render
    ####################
    bpy.context.scene.render.filepath = os.path.join(output_dir, output_fname)
    # bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.render.render(write_still=True)

    # if object_path.endswith('_rot.obj'):
    #     pass
    # else:
    #     # Overlay image
    #     filepath = os.path.join(output_dir, output_file)
    #     overlay_img = overlay_smooth(filepath.replace('_render.png', '.jpg'), filepath)
    #     save_image(filepath.replace('_render.png', '_overlay.png'), overlay_img)

    # Delete last selected object from scene
    object.select_set(True)
    bpy.ops.object.delete()


##############################################################################
# Main
##############################################################################

if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    parser.add_argument('-i',
                        '--inp',
                        type=str,
                        required=True,
                        help='input directory')
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        default=None,
                        help='output directory')
    parser.add_argument('-w',
                        '--wireframe',
                        action='store_true',
                        help='draws quad wireframe')
    parser.add_argument('-t',
                        '--thickness',
                        type=float,
                        default=0.15,
                        help='wireframe thickness')
    parser.add_argument('-c',
                        '--color',
                        type=str,
                        default='turkuaz',
                        help='mesh color')
    parser.add_argument('-s',
                        '--size',
                        type=int,
                        default=720,
                        help='image size')
    parser.add_argument('--sideview',
                        action='store_true',
                        help='flag to render side view meshes')
    parser.add_argument('--turntable',
                        action='store_true',
                        help='render with turntable')
    args = parser.parse_args()

    # argv = sys.argv
    # argv = argv[argv.index("--") + 1:]  # get all args after "--"

    print('Input arguments:', args)  # --> ['example', 'args', '123']

    if args.turntable:
        assert args.inp.endswith(
            '.obj'
        ) is True, 'Single obj file should be provided for turntable.'

    if args.inp.endswith('.obj'):
        print('Processing a single file')
        input_file = args.inp
        input_file = os.path.abspath(input_file)
        filelist = [os.path.basename(args.inp)]
        input_dir = input_file.replace(os.path.basename(args.inp), '')
        output_dir = input_dir
    else:
        input_dir = args.inp
        output_dir = args.out if args.out else input_dir.replace(
            'mesh_output', 'blender_output')
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        # Process data in directory
        if args.sideview:
            filelist = [
                x for x in sorted(os.listdir(input_dir)) if x.endswith('.obj')
            ]
        else:
            filelist = [
                x for x in sorted(os.listdir(input_dir))
                if x.endswith('.obj') and not x.endswith('_rot.obj')
            ]

    wireframe = args.wireframe
    debug = False

    # Render setup
    scene = bpy.data.scenes['Scene']
    scene.render.use_freestyle = wireframe
    scene.render.line_thickness = args.thickness

    # Change mesh color
    mc = get_colors(args.color) / 255.
    bpy.data.materials['Body'].node_tree.nodes["Principled BSDF"].inputs[
        0].default_value = (*mc, 1)

    img_size = args.size

    print('Num of files to be processed', len(filelist))

    if args.turntable:
        for idx, input_file in enumerate(filelist):
            print(input_file)

            mesh_fn = os.path.join(input_dir, input_file)
            np_path = mesh_fn.replace('.obj', '.npy')

            out_dir = mesh_fn.replace('.obj', '')
            # if os.path.isdir(out_dir):
            #     print('Results are already rendered!')
            #     pass
            # else:
            os.makedirs(out_dir, exist_ok=True)

            frame_idx = 0
            for rot in range(0, 360, 4):
                output_file = os.path.join(out_dir, f'{frame_idx:03d}.png')
                render_turntable(
                    mesh_fn,
                    output_fname=output_file,
                    numpy_path=np_path,
                    wireframe=False,
                    quads=True,
                    width=img_size,
                    height=img_size,
                    rotation_angle=rot,
                )
                frame_idx += 1

            # This saves the highest quality images
            cmd = f'ffmpeg -y -framerate 30 -i {out_dir}/%03d.png -c:v copy {mesh_fn.replace(".obj", "")}.mp4'
            os.system(cmd)
    else:
        for idx, input_file in enumerate(filelist):
            print(input_file)

            mesh_fn = os.path.join(input_dir, input_file)

            if input_file.endswith('_rot.obj'):
                np_path = '_'.join(mesh_fn.split('_')[:-2]) + '.npy'
            else:
                np_path = mesh_fn.replace('.obj', '.npy')

            process_data(
                mesh_fn,
                output_dir,
                numpy_path=np_path,
                wireframe=False,
                quads=True,
                width=img_size,
                height=img_size,
            )

            if debug:
                break
