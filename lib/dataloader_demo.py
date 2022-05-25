import argparse
from lib.common.config import get_cfg_defaults
from lib.dataset.PIFuDataset import PIFuDataset

args = get_cfg_defaults()
args.merge_from_file("./configs/train/icon-filter.yaml")

# loading cfg file
parser = argparse.ArgumentParser()
parser.add_argument('-v',
                    '--show',
                    action='store_true',
                    help='vis sampler 3D')
parser.add_argument('-s',
                    '--speed',
                    action='store_true',
                    help='vis sampler 3D')
parser.add_argument('-l',
                    '--list',
                    action='store_true',
                    help='vis sampler 3D')
args_c = parser.parse_args()

dataset = PIFuDataset(args, split='train', vis=args_c.show)
print(f"Number of subjects :{len(dataset.subject_list)}")
data_dict = dataset[0]

if args_c.list:
    for k in data_dict.keys():
        if not hasattr(data_dict[k], "shape"):
            print(f"{k}: {data_dict[k]}")
        else:
            print(f"{k}: {data_dict[k].shape}")

if args_c.show:
    # for item in dataset:
    item = dataset[0]
    dataset.visualize_sampling3D(item, mode='occ')

if args_c.speed:
    # original: 2 it/s
    # smpl online compute: 2 it/s
    # normal online compute: 1.5 it/s
    from tqdm import tqdm
    for item in tqdm(dataset):
        # pass
        for k in item.keys():
            if 'voxel' in k:
                if not hasattr(item[k], "shape"):
                    print(f"{k}: {item[k]}")
                else:
                    print(f"{k}: {item[k].shape}")
        print("--------------------")
