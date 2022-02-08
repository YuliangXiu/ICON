import argparse


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        gen = self.parser.add_argument_group('General')
        gen.add_argument(
            '--resume',
            dest='resume',
            default=False,
            action='store_true',
            help='Resume from checkpoint (Use latest checkpoint by default')

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir',
                        default='logs',
                        help='Directory to store logs')
        io.add_argument(
            '--pretrained_checkpoint',
            default=None,
            help='Load a pretrained checkpoint at the beginning training')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs',
                           type=int,
                           default=200,
                           help='Total number of training epochs')
        train.add_argument('--regressor',
                           type=str,
                           choices=['hmr', 'pymaf_net'],
                           default='pymaf_net',
                           help='Name of the SMPL regressor.')
        train.add_argument('--cfg_file',
                           type=str,
                           default='./configs/pymaf_config.yaml',
                           help='config file path for PyMAF.')
        train.add_argument(
            '--img_res',
            type=int,
            default=224,
            help=
            'Rescale bounding boxes to size [img_res, img_res] before feeding them in the network'
        )
        train.add_argument(
            '--rot_factor',
            type=float,
            default=30,
            help='Random rotation in the range [-rot_factor, rot_factor]')
        train.add_argument(
            '--noise_factor',
            type=float,
            default=0.4,
            help=
            'Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]'
        )
        train.add_argument(
            '--scale_factor',
            type=float,
            default=0.25,
            help=
            'Rescale bounding boxes by a factor of [1-scale_factor,1+scale_factor]'
        )
        train.add_argument(
            '--openpose_train_weight',
            default=0.,
            help='Weight for OpenPose keypoints during training')
        train.add_argument('--gt_train_weight',
                           default=1.,
                           help='Weight for GT keypoints during training')
        train.add_argument('--eval_dataset',
                           type=str,
                           default='h36m-p2-mosh',
                           help='Name of the evaluation dataset.')
        train.add_argument('--single_dataset',
                           default=False,
                           action='store_true',
                           help='Use a single dataset')
        train.add_argument('--single_dataname',
                           type=str,
                           default='h36m',
                           help='Name of the single dataset.')
        train.add_argument('--eval_pve',
                           default=False,
                           action='store_true',
                           help='evaluate PVE')
        train.add_argument('--overwrite',
                           default=False,
                           action='store_true',
                           help='overwrite the latest checkpoint')

        train.add_argument('--distributed',
                           action='store_true',
                           help='Use distributed training')
        train.add_argument('--dist_backend',
                           default='nccl',
                           type=str,
                           help='distributed backend')
        train.add_argument('--dist_url',
                           default='tcp://127.0.0.1:10356',
                           type=str,
                           help='url used to set up distributed training')
        train.add_argument('--world_size',
                           default=1,
                           type=int,
                           help='number of nodes for distributed training')
        train.add_argument("--local_rank", default=0, type=int)
        train.add_argument('--rank',
                           default=0,
                           type=int,
                           help='node rank for distributed training')
        train.add_argument(
            '--multiprocessing_distributed',
            action='store_true',
            help='Use multi-processing distributed training to launch '
            'N processes per node, which has N GPUs. This is the '
            'fastest way to use PyTorch for either single node or '
            'multi node data parallel training')

        misc = self.parser.add_argument_group('Misc Options')
        misc.add_argument('--misc',
                          help="Modify config options using the command-line",
                          default=None,
                          nargs=argparse.REMAINDER)
        return

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        self.save_dump()
        return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/args.json.
        """
        pass
