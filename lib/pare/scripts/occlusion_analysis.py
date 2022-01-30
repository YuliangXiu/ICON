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
import cv2
import math
import time
import torch
import shutil
import joblib
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from loguru import logger
import matplotlib.pylab as plt

from pare.core.trainer import PARETrainer
from pare.utils.train_utils import load_pretrained_model
from pare.core.config import run_grid_search_experiments
from pare.utils.kp_utils import get_common_joint_names


def get_occluded_imgs(img, occ_size, occ_pixel, occ_stride):

    img_size = int(img.shape[-1])
    # Define number of occlusions in both dimensions
    output_height = int(math.ceil((img_size - occ_size) / occ_stride + 1))
    output_width = int(math.ceil((img_size - occ_size) / occ_stride + 1))

    occ_img_list = []

    idx_dict = {}
    c = 0
    for h in range(output_height):
        for w in range(output_width):
            # Occluder window:
            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(img_size, h_start + occ_size)
            w_end = min(img_size, w_start + occ_size)

            # Getting the image copy, applying the occluding window:
            occ_image = img.clone()
            occ_image[:, :, h_start:h_end, w_start:w_end] = occ_pixel
            occ_img_list.append(occ_image)

            idx_dict[c] = (h, w)
            c += 1

    return torch.stack(occ_img_list, dim=0), idx_dict, output_height


def visualize_grid(image, heatmap, imgname=None, res_dict=None):

    image = image * torch.tensor([0.229, 0.224, 0.225],
                                 device=image.device).reshape(1, 3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406],
                                 device=image.device).reshape(1, 3, 1, 1)
    image = np.transpose(image.cpu().numpy(), (0, 2, 3, 1))[0]

    orig_heatmap = heatmap.copy()
    # heatmap = resize(heatmap, image.shape)
    heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]),
                         interpolation=cv2.INTER_CUBIC)

    heatmap = (heatmap - np.min(heatmap)) / np.ptp(
        heatmap)  # normalize between [0,1]

    title = ''
    if imgname:
        title += '/'.join(imgname.split('/')[-2:]) + '\n'

    if res_dict:
        title += f' err: {res_dict["mpjpe"]*1000:.2f}mm'
        title += f' r_err: {res_dict["pampjpe"]*1000:.2f}mm'

    w, h = 7, 2
    f, axarr = plt.subplots(h, w)
    f.set_size_inches((w * 3, h * 3))

    f.suptitle(title)
    joint_names = get_common_joint_names()

    for jid in range(len(joint_names)):
        axarr[jid // w, jid % w].axis('off')
        axarr[jid // w, jid % w].set_title(
            f'{joint_names[jid]} \n'
            f'min: {orig_heatmap[:,:,jid].min()*1000:.2f} '
            f'max: {orig_heatmap[:,:,jid].max()*1000:.2f}')
        axarr[jid // w, jid % w].imshow(image)

        axarr[jid // w, jid % w].imshow(heatmap[:, :, jid],
                                        alpha=.5,
                                        cmap='jet',
                                        interpolation='none')

    f.set_tight_layout(tight=True)


def run_dataset(args, hparams):

    if args.ckpt is not None:
        logger.info(f'Pretrained checkpoint is \"{args.ckpt}\"')
        hparams.TRAINING.PRETRAINED_LIT = args.ckpt

    if args.dataset is not None:
        logger.info(f'Test dataset is \"{args.dataset}\"')
        hparams.DATASET.VAL_DS = args.dataset

    if args.batch_size is not None:
        logger.info(f'Testing batch size \"{args.batch_size}\"')
        hparams.DATASET.BATCH_SIZE = args.batch_size

    hparams.RUN_TEST = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')

    model = PARETrainer(hparams=hparams).to(device)
    model = model.eval()

    val_images_errors = []

    if hparams.TRAINING.PRETRAINED_LIT is not None:
        logger.warning(
            f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True)

    dataloader = model.val_dataloader()[0]

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % args.save_freq != 0:
            continue

        logger.info(
            f'Processing {batch_idx} / {len(dataloader)} "{batch["imgname"]}"')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].to(device)

        occluded_images, idx_dict, output_size = get_occluded_imgs(
            batch['img'],
            occ_size=args.occ_size,
            occ_pixel=args.pixel,
            occ_stride=args.stride)
        ratio = hparams.DATASET.RENDER_RES / hparams.DATASET.IMG_RES
        occluded_images_disp, idx_dict, output_size = get_occluded_imgs(
            batch['disp_img'],
            occ_size=int(round(args.occ_size * ratio)),
            occ_pixel=args.pixel,
            occ_stride=int(round(args.stride * ratio)),
        )

        mpjpe_heatmap = np.zeros((output_size, output_size, 14))
        pampjpe_heatmap = np.zeros((output_size, output_size, 14))

        orig_image = batch['disp_img']

        model.hparams.TESTING.SAVE_MESHES = False
        orig_res_dict = model.validation_step(batch,
                                              batch_idx,
                                              dataloader_nb=0,
                                              vis=True,
                                              save=True)

        val_images_errors.append(
            [orig_res_dict['mpjpe'], orig_res_dict['pampjpe']])

        save_dir = os.path.join(hparams.LOG_DIR, 'output_images',
                                f'result_00_{batch_idx:05d}_occlusion')
        os.makedirs(save_dir, exist_ok=True)

        for occ_img_idx in tqdm(range(occluded_images.shape[0])):
            batch['img'] = occluded_images[occ_img_idx]
            batch['disp_img'] = occluded_images_disp[occ_img_idx]

            model.hparams.TESTING.SAVE_MESHES = False
            result_dict = model.validation_step(
                batch,
                occ_img_idx,
                dataloader_nb=0,
                vis=True,
                save=False,
            )

            cv2.imwrite(
                os.path.join(save_dir, f'result_{occ_img_idx:05d}.jpg'),
                result_dict['vis_image'])

            mpjpe_heatmap[idx_dict[occ_img_idx]] = result_dict['per_mpjpe'][0]
            pampjpe_heatmap[
                idx_dict[occ_img_idx]] = result_dict['per_pampjpe'][0]

        command = [
            'ffmpeg', '-y', '-framerate', '15', '-i',
            f'{save_dir}/result_%05d.jpg', '-c:v', 'libx264', '-profile:v',
            'high', '-crf', '20', '-pix_fmt', 'yuv420p',
            os.path.join(hparams.LOG_DIR, 'output_images',
                         f'result_00_{batch_idx:05d}.mp4')
        ]
        logger.info(f'Running {"".join(command)}')
        subprocess.call(command)
        shutil.rmtree(save_dir)

        fig = plt.figure()
        visualize_grid(orig_image, mpjpe_heatmap, res_dict=orig_res_dict)
        plt.savefig(
            os.path.join(hparams.LOG_DIR, 'output_images',
                         f'result_00_{batch_idx:05d}_mpjpe_hm.png'))
        # plt.show()

        visualize_grid(orig_image, pampjpe_heatmap, res_dict=orig_res_dict)
        plt.savefig(
            os.path.join(hparams.LOG_DIR, 'output_images',
                         f'result_00_{batch_idx:05d}_pampjpe_hm.png'))
        # plt.show()

        plt.close(fig)

        orig_res_dict['mpjpe_heatmap'] = mpjpe_heatmap
        orig_res_dict['pampjpe_heatmap'] = pampjpe_heatmap
        orig_res_dict['imgname'] = batch['imgname']

        del orig_res_dict['vis_image']

        joblib.dump(
            value=orig_res_dict,
            filename=os.path.join(hparams.LOG_DIR, 'output_images',
                                  f'result_00_{batch_idx:05d}.pkl'),
        )

        plt.close(fig)
        plt.clf()
        plt.cla()
        plt.close('all')

    save_path = os.path.join(hparams.LOG_DIR, 'val_images_error.npy')
    logger.info(f'Saving the errors of images {save_path}')
    np.save(save_path, np.asarray(val_images_errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--ckpt', type=str,
                        default=None)  # Path of the saved pre-trained model
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_freq', type=int,
                        default='100')  # save frequency
    parser.add_argument('--dataset', type=str,
                        default='3dpw-all_3doh')  # Path of the input image
    parser.add_argument('--occ_size', type=int,
                        default='40')  # Size of occluding window
    parser.add_argument('--pixel', type=int,
                        default='0')  # Occluding window - pixel values
    parser.add_argument('--stride', type=int, default='40')  # Occlusion Stride

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=0,
        cfg_file=args.cfg,
        bid=300,
        use_cluster=False,
        memory=16000,
        script='occlusion_analysis.py',
    )

    run_dataset(args, hparams)
