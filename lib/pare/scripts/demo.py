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
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import cv2
import time
import joblib
import argparse
from loguru import logger

sys.path.append('.')
from pare.core.tester import PARETester
from pare.utils.demo_utils import (
    download_youtube_clip,
    video_to_images,
    images_to_video,
)

CFG = 'data/pare/checkpoints/pare_w_3dpw_config.yaml'
CKPT = 'data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt'
MIN_NUM_FRAMES = 0


def main(args):

    demo_mode = args.mode

    if demo_mode == 'video':
        video_file = args.vid_file

        # ========= [Optional] download the youtube video ========= #
        if video_file.startswith('https://www.youtube.com'):
            logger.info(f'Downloading YouTube video \"{video_file}\"')
            video_file = download_youtube_clip(video_file, '/tmp')

            if video_file is None:
                exit('Youtube url is not valid!')

            logger.info(
                f'YouTube Video has been downloaded to {video_file}...')

        if not os.path.isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')

        output_path = os.path.join(
            args.output_folder,
            os.path.basename(video_file).replace('.mp4', '_' + args.exp))
        os.makedirs(output_path, exist_ok=True)

        if os.path.isdir(os.path.join(output_path, 'tmp_images')):
            input_image_folder = os.path.join(output_path, 'tmp_images')
            logger.info(
                f'Frames are already extracted in \"{input_image_folder}\"')
            num_frames = len(os.listdir(input_image_folder))
            img_shape = cv2.imread(
                os.path.join(input_image_folder, '000001.png')).shape
        else:
            input_image_folder, num_frames, img_shape = video_to_images(
                video_file,
                img_folder=os.path.join(output_path, 'tmp_images'),
                return_info=True)
        output_img_folder = f'{input_image_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)
    elif demo_mode == 'folder':
        args.tracker_batch_size = 1
        input_image_folder = args.image_folder
        output_path = os.path.join(
            args.output_folder,
            input_image_folder.rstrip('/').split('/')[-1] + '_' + args.exp)
        os.makedirs(output_path, exist_ok=True)

        output_img_folder = os.path.join(output_path, 'pare_results')
        os.makedirs(output_img_folder, exist_ok=True)

        num_frames = len(os.listdir(input_image_folder))
    elif demo_mode == 'webcam':
        logger.error('Webcam demo is not implemented!..')
        raise NotImplementedError
    else:
        raise ValueError(f'{demo_mode} is not a valid demo mode.')

    logger.add(
        os.path.join(output_path, 'demo.log'),
        level='INFO',
        colorize=False,
    )
    logger.info(f'Demo options: \n {args}')

    tester = PARETester(args)

    total_time = time.time()
    if args.mode == 'video':
        logger.info(f'Input video number of frames {num_frames}')
        orig_height, orig_width = img_shape[:2]
        total_time = time.time()
        tracking_results = tester.run_tracking(video_file, input_image_folder)
        pare_time = time.time()
        pare_results = tester.run_on_video(tracking_results,
                                           input_image_folder, orig_width,
                                           orig_height)
        end = time.time()

        fps = num_frames / (end - pare_time)

        del tester.model

        logger.info(f'PARE FPS: {fps:.2f}')
        total_time = time.time() - total_time
        logger.info(
            f'Total time spent: {total_time:.2f} seconds (including model loading time).'
        )
        logger.info(
            f'Total FPS (including model loading time): {num_frames / total_time:.2f}.'
        )

        if not args.no_save:
            logger.info(
                f'Saving output results to \"{os.path.join(output_path, "pare_output.pkl")}\".'
            )
            joblib.dump(pare_results,
                        os.path.join(output_path, "pare_output.pkl"))

        if not args.no_render:
            tester.render_results(pare_results, input_image_folder,
                                  output_img_folder, output_path, orig_width,
                                  orig_height, num_frames)

            # ========= Save rendered video ========= #
            vid_name = os.path.basename(video_file)
            save_name = f'{vid_name.replace(".mp4", "")}_{args.exp}_result.mp4'
            save_name = os.path.join(output_path, save_name)
            logger.info(f'Saving result video to {save_name}')
            images_to_video(img_folder=output_img_folder,
                            output_vid_file=save_name)

            # Save the input video as well
            images_to_video(img_folder=input_image_folder,
                            output_vid_file=os.path.join(
                                output_path, vid_name))
            # shutil.rmtree(output_img_folder)

        # shutil.rmtree(image_folder)
    elif args.mode == 'folder':
        logger.info(f'Number of input frames {num_frames}')

        total_time = time.time()
        detections = tester.run_detector(input_image_folder)
        pare_time = time.time()
        tester.run_on_image_folder(input_image_folder, detections, output_path,
                                   output_img_folder)
        end = time.time()

        fps = num_frames / (end - pare_time)

        del tester.model

        logger.info(f'PARE FPS: {fps:.2f}')
        total_time = time.time() - total_time
        logger.info(
            f'Total time spent: {total_time:.2f} seconds (including model loading time).'
        )
        logger.info(
            f'Total FPS (including model loading time): {num_frames / total_time:.2f}.'
        )

    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        type=str,
                        default=CFG,
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt',
                        type=str,
                        default=CKPT,
                        help='checkpoint path')

    parser.add_argument('--exp',
                        type=str,
                        default='',
                        help='short description of the experiment')

    parser.add_argument('--mode',
                        default='video',
                        choices=['video', 'folder', 'webcam'],
                        help='Demo type')

    parser.add_argument('--vid_file',
                        type=str,
                        help='input video path or youtube link')

    parser.add_argument('--image_folder', type=str, help='input image folder')

    parser.add_argument('--output_folder',
                        type=str,
                        default='logs/demo/demo_results',
                        help='output folder to write results')

    parser.add_argument(
        '--tracking_method',
        type=str,
        default='bbox',
        choices=['bbox', 'pose'],
        help=
        'tracking method to calculate the tracklet of a subject from the input video'
    )

    parser.add_argument('--detector',
                        type=str,
                        default='yolo',
                        choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size',
                        type=int,
                        default=416,
                        help='input image size for yolo detector')

    parser.add_argument(
        '--tracker_batch_size',
        type=int,
        default=12,
        help='batch size of object detector used for bbox tracking')

    parser.add_argument(
        '--staf_dir',
        type=str,
        default='/home/mkocabas/developments/openposetrack',
        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='batch size of PARE')

    parser.add_argument('--display',
                        action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--smooth',
                        action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument(
        '--min_cutoff',
        type=float,
        default=0.004,
        help='one euro filter min cutoff. '
        'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument(
        '--beta',
        type=float,
        default=1.0,
        help='one euro filter beta. '
        'Increasing the speed coefficient(beta) decreases speed lag.')

    parser.add_argument('--no_render',
                        action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--no_save',
                        action='store_true',
                        help='disable final save of output results.')

    parser.add_argument('--wireframe',
                        action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview',
                        action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--draw_keypoints',
                        action='store_true',
                        help='draw 2d keypoints on rendered image.')

    parser.add_argument('--save_obj',
                        action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()

    main(args)
