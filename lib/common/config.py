
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

from yacs.config import CfgNode as CN
import os

_C = CN(new_allowed=True)

# needed by trainer
_C.name = 'default'
_C.gpus = [0]
_C.test_gpus = [1]
_C.root = "../data/"
_C.ckpt_dir = '../data/ckpt/'
_C.resume_path = ''
_C.normal_path = ''
_C.corr_path = ''
_C.results_path = '../data/results/'
_C.projection_mode = 'orthogonal'
_C.num_views = 1
_C.sdf = False
_C.sdf_clip = 5.0

_C.lr_G = 1e-3
_C.lr_C = 1e-3
_C.lr_N = 2e-4
_C.weight_decay = 0.0
_C.momentum = 0.0
_C.optim = 'RMSprop'
_C.schedule = [5, 10, 15]
_C.gamma = 0.1

_C.overfit = False
_C.resume = False
_C.test_mode = False
_C.test_uv = False
_C.draw_geo_thres = 0.60
_C.num_sanity_val_steps = 2
_C.fast_dev = 0
_C.get_fit = False
_C.agora = False
_C.optim_cloth = False
_C.optim_body = False
_C.mcube_res = 256
_C.clean_mesh = True

_C.batch_size = 4
_C.num_threads = 8

_C.num_epoch = 10
_C.freq_plot = 0.01
_C.freq_show_train = 0.1
_C.freq_show_val = 0.2
_C.freq_eval = 0.5
_C.accu_grad_batch = 4

_C.test_items = ['sv', 'mv', 'mv-fusion', 'hybrid', 'dc-pred', 'gt']

_C.net = CN()
_C.net.gtype = 'HGPIFuNet'
_C.net.ctype = 'resnet18'
_C.net.classifierIMF = 'MultiSegClassifier'
_C.net.netIMF = 'resnet18'
_C.net.norm = 'group'
_C.net.norm_mlp = 'group'
_C.net.norm_color = 'group'
_C.net.hg_down = 'ave_pool'
_C.net.num_views = 1

# kernel_size, stride, dilation, padding

_C.net.conv1 = [7, 2, 1, 3]
_C.net.conv3x3 = [3, 1, 1, 1]

_C.net.num_stack = 4
_C.net.num_hourglass = 2
_C.net.hourglass_dim = 256
_C.net.voxel_dim = 32
_C.net.resnet_dim = 120
_C.net.mlp_dim = [320, 1024, 512, 256, 128, 1]
_C.net.mlp_dim_knn = [320, 1024, 512, 256, 128, 3]
_C.net.mlp_dim_color = [513, 1024, 512, 256, 128, 3]
_C.net.mlp_dim_multiseg = [1088, 2048, 1024, 500]
_C.net.res_layers = [2, 3, 4]
_C.net.filter_dim = 256
_C.net.smpl_dim = 3

_C.net.cly_dim = 3
_C.net.soft_dim = 64
_C.net.z_size = 200.0
_C.net.N_freqs = 10
_C.net.geo_w = 0.1
_C.net.norm_w = 0.1
_C.net.dc_w = 0.1
_C.net.C_cat_to_G = False

_C.net.skip_hourglass = True
_C.net.use_tanh = True
_C.net.soft_onehot = True
_C.net.no_residual = True
_C.net.use_attention = False

_C.net.prior_type = "sdf"
_C.net.smpl_feats = ['sdf', 'cmap', 'norm', 'vis']
_C.net.use_filter = True
_C.net.use_cc = False
_C.net.use_PE = False
_C.net.use_IGR = False
_C.net.in_geo = ()
_C.net.in_nml = ()

_C.dataset = CN()
_C.dataset.root = ''
_C.dataset.set_splits = [0.95, 0.04]
_C.dataset.types = [
    "3dpeople", "axyz", "renderpeople", "renderpeople_p27", "humanalloy"
]
_C.dataset.scales = [1.0, 100.0, 1.0, 1.0, 100.0 / 39.37]
_C.dataset.rp_type = "pifu900"
_C.dataset.th_type = 'train'
_C.dataset.input_size = 512
_C.dataset.rotation_num = 3
_C.dataset.num_precomp = 10  # Number of segmentation classifiers
_C.dataset.num_multiseg = 500  # Number of categories per classifier
_C.dataset.num_knn = 10  # for loss/error
_C.dataset.num_knn_dis = 20  # for accuracy
_C.dataset.num_verts_max = 20000
_C.dataset.zray_type = False
_C.dataset.online_smpl = False
_C.dataset.noise_type = ['z-trans', 'pose', 'beta']
_C.dataset.noise_scale = [0.0, 0.0, 0.0]
_C.dataset.num_sample_geo = 10000
_C.dataset.num_sample_color = 0
_C.dataset.num_sample_seg = 0
_C.dataset.num_sample_knn = 10000

_C.dataset.sigma_geo = 5.0
_C.dataset.sigma_color = 0.10
_C.dataset.sigma_seg = 0.10
_C.dataset.thickness_threshold = 20.0
_C.dataset.ray_sample_num = 2
_C.dataset.semantic_p = False
_C.dataset.remove_outlier = False

_C.dataset.train_bsize = 1.0
_C.dataset.val_bsize = 1.0
_C.dataset.test_bsize = 1.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C  # users can `from config import cfg`

# cfg = get_cfg_defaults()
# cfg.merge_from_file('../configs/example.yaml')

# # Now override from a list (opts could come from the command line)
# opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
# cfg.merge_from_list(opts)


def update_cfg(cfg_file):
    # cfg = get_cfg_defaults()
    _C.merge_from_file(cfg_file)
    # return cfg.clone()
    return _C


def parse_args(args):
    cfg_file = args.cfg_file
    if args.cfg_file is not None:
        cfg = update_cfg(args.cfg_file)
    else:
        cfg = get_cfg_defaults()

    # if args.misc is not None:
    #     cfg.merge_from_list(args.misc)

    return cfg


def parse_args_extend(args):
    if args.resume:
        if not os.path.exists(args.log_dir):
            raise ValueError(
                'Experiment are set to resume mode, but log directory does not exist.'
            )

        # load log's cfg
        cfg_file = os.path.join(args.log_dir, 'cfg.yaml')
        cfg = update_cfg(cfg_file)

        if args.misc is not None:
            cfg.merge_from_list(args.misc)
    else:
        parse_args(args)
