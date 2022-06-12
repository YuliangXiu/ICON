'''
borrowed from https://github.com/vchoutas/expose/blob/master/expose/models/backbone/hrnet.py
'''

import os.path as osp
import torch
import torch.nn as nn

from torchvision.models.resnet import Bottleneck, BasicBlock

BN_MOMENTUM = 0.1


def load_HRNet(pretrained=False):
    hr_net_cfg_dict = {
        'use_old_impl': False,
        'pretrained_layers': ['*'],
        'stage1':
        {'num_modules': 1, 'num_branches': 1, 'num_blocks': [4], 'num_channels': [
            64], 'block': 'BOTTLENECK', 'fuse_method': 'SUM'},
        'stage2':
        {'num_modules': 1, 'num_branches': 2, 'num_blocks': [
            4, 4], 'num_channels': [48, 96], 'block': 'BASIC', 'fuse_method': 'SUM'},
        'stage3':
        {'num_modules': 4, 'num_branches': 3, 'num_blocks': [4, 4, 4], 'num_channels': [
            48, 96, 192], 'block': 'BASIC', 'fuse_method': 'SUM'},
        'stage4':
        {'num_modules': 3, 'num_branches': 4, 'num_blocks': [4, 4, 4, 4], 'num_channels': [
            48, 96, 192, 384], 'block': 'BASIC', 'fuse_method': 'SUM'}
    }
    hr_net_cfg = hr_net_cfg_dict
    model = HighResolutionNet(hr_net_cfg)

    return model


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        super(HighResolutionNet, self).__init__()
        use_old_impl = cfg.get('use_old_impl')
        self.use_old_impl = use_old_impl

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg.get('stage1', {})
        num_channels = self.stage1_cfg['num_channels'][0]
        block = blocks_dict[self.stage1_cfg['block']]
        num_blocks = self.stage1_cfg['num_blocks'][0]
        self.layer1 = self._make_layer(block, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg.get('stage2', {})
        num_channels = self.stage2_cfg.get('num_channels', (32, 64))
        block = blocks_dict[self.stage2_cfg.get('block')]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        stage2_num_channels = num_channels
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg.get('stage3')
        num_channels = self.stage3_cfg['num_channels']
        block = blocks_dict[self.stage3_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        stage3_num_channels = num_channels
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg.get('stage4')
        num_channels = self.stage4_cfg['num_channels']
        block = blocks_dict[self.stage4_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        stage_4_out_channels = num_channels

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels,
            multi_scale_output=not self.use_old_impl)
        stage4_num_channels = num_channels

        self.output_channels_dim = pre_stage_channels

        self.pretrained_layers = cfg['pretrained_layers']
        self.init_weights()

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        if use_old_impl:
            in_dims = (2 ** 2 * stage2_num_channels[-1] +
                       2 ** 1 * stage3_num_channels[-1] +
                       stage_4_out_channels[-1]
                       )
        else:
            # TODO: Replace with parameters
            in_dims = 4 * 384
            self.subsample_4 = self._make_subsample_layer(
                in_channels=stage4_num_channels[0], num_layers=3)

        self.subsample_3 = self._make_subsample_layer(
            in_channels=stage2_num_channels[-1], num_layers=2)
        self.subsample_2 = self._make_subsample_layer(
            in_channels=stage3_num_channels[-1], num_layers=1)
        self.conv_layers = self._make_conv_layer(
            in_channels=in_dims, num_layers=5)

    def get_output_dim(self):
        base_output = {
            f'layer{idx + 1}': val
            for idx, val in enumerate(self.output_channels_dim)
        }
        output = base_output.copy()
        for key in base_output:
            output[f'{key}_avg_pooling'] = output[key]
        output['concat'] = 2048
        return output

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_layer(self, in_channels=2048, num_layers=3,
                         num_filters=2048, stride=1):

        layers = []
        for i in range(num_layers):

            downsample = nn.Conv2d(in_channels, num_filters, stride=1,
                                   kernel_size=1, bias=False)
            layers.append(Bottleneck(in_channels, num_filters // 4,
                                     downsample=downsample))
            in_channels = num_filters

        return nn.Sequential(*layers)

    def _make_subsample_layer(self, in_channels=96, num_layers=3, stride=2):

        layers = []
        for i in range(num_layers):

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1))
            in_channels = 2 * in_channels
            layers.append(nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True, log=False):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = blocks_dict[layer_config['block']]
        fuse_method = layer_config['fuse_method']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            modules[-1].log = log
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['num_branches']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['num_branches']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        if not self.use_old_impl:
            y_list = self.stage4(x_list)

        output = {}
        for idx, x in enumerate(y_list):
            output[f'layer{idx + 1}'] = x

        feat_list = []
        if self.use_old_impl:
            x3 = self.subsample_3(x_list[1])
            x2 = self.subsample_2(x_list[2])
            x1 = x_list[3]
            feat_list = [x3, x2, x1]
        else:
            x4 = self.subsample_4(y_list[0])
            x3 = self.subsample_3(y_list[1])
            x2 = self.subsample_2(y_list[2])
            x1 = y_list[3]
            feat_list = [x4, x3, x2, x1]

        xf = self.conv_layers(torch.cat(feat_list, dim=1))
        xf = xf.mean(dim=(2, 3))
        xf = xf.view(xf.size(0), -1)
        output['concat'] = xf
        #  y_list = self.stage4(x_list)
        #  output['stage4'] = y_list[0]
        #  output['stage4_avg_pooling'] = self.avg_pooling(y_list[0]).view(
        #  *y_list[0].shape[:2])

        #  concat_outputs = y_list + x_list
        #  output['concat'] = torch.cat([
        #  self.avg_pooling(tensor).view(*tensor.shape[:2])
        #  for tensor in concat_outputs],
        #  dim=1)

        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def load_weights(self, pretrained=''):
        pretrained = osp.expandvars(pretrained)
        if osp.isfile(pretrained):
            pretrained_state_dict = torch.load(
                pretrained, map_location=torch.device("cpu"))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if (name.split('.')[0] in self.pretrained_layers or
                        self.pretrained_layers[0] == '*'):
                    need_init_state_dict[name] = m
            missing, unexpected = self.load_state_dict(
                need_init_state_dict, strict=False)
        elif pretrained:
            raise ValueError('{} is not exist!'.format(pretrained))
