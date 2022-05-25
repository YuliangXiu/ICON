'''
Default config for PIXIE
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_pixie_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
cfg.pixie_dir = abs_pixie_dir
cfg.device = 'cuda'
cfg.device_id = '0'
cfg.pretrained_modelpath = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'pixie_model.tar')
# smplx parameter settings
cfg.params = CN()
cfg.params.body_list = ['body_cam',
                        'global_pose', 'partbody_pose', 'neck_pose']
cfg.params.head_list = ['head_cam', 'tex', 'light']
cfg.params.head_share_list = ['shape', 'exp', 'head_pose', 'jaw_pose']
cfg.params.hand_list = ['hand_cam']
cfg.params.hand_share_list = ['right_wrist_pose',
                              'right_hand_pose']  # only for right hand

# ---------------------------------------------------------------------------- #
# Options for Body model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'SMPL_X_template_FLAME_uv.obj')
cfg.model.topology_smplxtex_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'smplx_tex.obj')
cfg.model.topology_smplx_hand_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'smplx_hand.obj')
cfg.model.smplx_model_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'SMPLX_NEUTRAL_2020.npz')
cfg.model.face_mask_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'uv_face_mask.png')
cfg.model.face_eye_mask_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'uv_face_eye_mask.png')
cfg.model.tex_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'FLAME_albedo_from_BFM.npz')
cfg.model.extra_joint_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'smplx_extra_joints.yaml')
cfg.model.j14_regressor_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'SMPLX_to_J14.pkl')
cfg.model.flame2smplx_cached_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'flame2smplx_tex_1024.npy')
cfg.model.smplx_tex_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'smplx_tex.png')
cfg.model.mano_ids_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'MANO_SMPLX_vertex_ids.pkl')
cfg.model.flame_ids_path = os.path.join(
    cfg.pixie_dir, 'data/pixie_data', 'SMPL-X__FLAME_vertex_ids.npy')
cfg.model.uv_size = 256
cfg.model.n_shape = 200
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_body_cam = 3
cfg.model.n_head_cam = 3
cfg.model.n_hand_cam = 3
cfg.model.tex_type = 'BFM'  # BFM, FLAME, albedoMM
cfg.model.uvtex_type = 'SMPLX'  # FLAME or SMPLX
cfg.model.use_tex = False  # whether to use flame texture model
cfg.model.flame_tex_path = ''

# pose
cfg.model.n_global_pose = 3*2
cfg.model.n_head_pose = 3*2
cfg.model.n_neck_pose = 3*2
cfg.model.n_jaw_pose = 3  # euler angle
cfg.model.n_body_pose = 21*3*2
cfg.model.n_partbody_pose = (21-4)*3*2
cfg.model.n_left_hand_pose = 15*3*2
cfg.model.n_right_hand_pose = 15*3*2
cfg.model.n_left_wrist_pose = 1*3*2
cfg.model.n_right_wrist_pose = 1*3*2
cfg.model.n_light = 27
cfg.model.check_pose = True

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.source = ['body', 'head', 'hand']

# head/face dataset
cfg.dataset.head = CN()
cfg.dataset.head.batch_size = 24
cfg.dataset.head.num_workers = 2
cfg.dataset.head.from_body = True
cfg.dataset.head.image_size = 224
cfg.dataset.head.image_hd_size = 224
cfg.dataset.head.scale_min = 1.8
cfg.dataset.head.scale_max = 2.2
cfg.dataset.head.trans_scale = 0.3
# body datset
cfg.dataset.body = CN()
cfg.dataset.body.batch_size = 24
cfg.dataset.body.num_workers = 2
cfg.dataset.body.image_size = 224
cfg.dataset.body.image_hd_size = 1024
cfg.dataset.body.use_hd = True
# hand datset
cfg.dataset.hand = CN()
cfg.dataset.hand.batch_size = 24
cfg.dataset.hand.num_workers = 2
cfg.dataset.hand.image_size = 224
cfg.dataset.hand.image_hd_size = 512
cfg.dataset.hand.scale_min = 2.2
cfg.dataset.hand.scale_max = 2.6
cfg.dataset.hand.trans_scale = 0.4

# ---------------------------------------------------------------------------- #
# Options for Network
# ---------------------------------------------------------------------------- #
cfg.network = CN()
cfg.network.encoder = CN()
cfg.network.encoder.body = CN()
cfg.network.encoder.body.type = 'hrnet'
cfg.network.encoder.head = CN()
cfg.network.encoder.head.type = 'resnet50'
cfg.network.encoder.hand = CN()
cfg.network.encoder.hand.type = 'resnet50'

cfg.network.regressor = CN()
cfg.network.regressor.head_share = CN()
cfg.network.regressor.head_share.type = 'mlp'
cfg.network.regressor.head_share.channels = [1024, 1024]
cfg.network.regressor.hand_share = CN()
cfg.network.regressor.hand_share.type = 'mlp'
cfg.network.regressor.hand_share.channels = [1024, 1024]
cfg.network.regressor.body = CN()
cfg.network.regressor.body.type = 'mlp'
cfg.network.regressor.body.channels = [1024]
cfg.network.regressor.head = CN()
cfg.network.regressor.head.type = 'mlp'
cfg.network.regressor.head.channels = [1024]
cfg.network.regressor.hand = CN()
cfg.network.regressor.hand.type = 'mlp'
cfg.network.regressor.hand.channels = [1024]

cfg.network.extractor = CN()
cfg.network.extractor.head_share = CN()
cfg.network.extractor.head_share.type = 'mlp'
cfg.network.extractor.head_share.channels = []
cfg.network.extractor.left_hand_share = CN()
cfg.network.extractor.left_hand_share.type = 'mlp'
cfg.network.extractor.left_hand_share.channels = []
cfg.network.extractor.right_hand_share = CN()
cfg.network.extractor.right_hand_share.type = 'mlp'
cfg.network.extractor.right_hand_share.channels = []

cfg.network.moderator = CN()
cfg.network.moderator.head_share = CN()
cfg.network.moderator.head_share.detach_inputs = False
cfg.network.moderator.head_share.detach_feature = False
cfg.network.moderator.head_share.type = 'temp-softmax'
cfg.network.moderator.head_share.channels = [1024, 1024]
cfg.network.moderator.head_share.reduction = 4
cfg.network.moderator.head_share.scale_type = 'scalars'
cfg.network.moderator.head_share.scale_init = 1.0
cfg.network.moderator.hand_share = CN()
cfg.network.moderator.hand_share.detach_inputs = False
cfg.network.moderator.hand_share.detach_feature = False
cfg.network.moderator.hand_share.type = 'temp-softmax'
cfg.network.moderator.hand_share.channels = [1024, 1024]
cfg.network.moderator.hand_share.reduction = 4
cfg.network.moderator.hand_share.scale_type = 'scalars'
cfg.network.moderator.hand_share.scale_init = 0.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg, cfg_file):
    # cfg.merge_from_file(cfg_file, allow_unsafe=True)
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file
    return cfg
