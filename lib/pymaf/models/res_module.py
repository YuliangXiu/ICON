# code brought in part from https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from lib.pymaf.core.cfgs import cfg

import logging

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1, bias=False, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes * groups,
                     out_planes * groups,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=bias,
                     groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes * groups, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes * groups, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes * groups,
                               planes * groups,
                               kernel_size=1,
                               bias=False,
                               groups=groups)
        self.bn1 = nn.BatchNorm2d(planes * groups, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes * groups,
                               planes * groups,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False,
                               groups=groups)
        self.bn2 = nn.BatchNorm2d(planes * groups, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes * groups,
                               planes * self.expansion * groups,
                               kernel_size=1,
                               bias=False,
                               groups=groups)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion * groups,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


class IUV_predict_layer(nn.Module):
    def __init__(self,
                 feat_dim=256,
                 final_cov_k=3,
                 part_out_dim=25,
                 with_uv=True):
        super().__init__()

        self.with_uv = with_uv
        if self.with_uv:
            self.predict_u = nn.Conv2d(in_channels=feat_dim,
                                       out_channels=25,
                                       kernel_size=final_cov_k,
                                       stride=1,
                                       padding=1 if final_cov_k == 3 else 0)

            self.predict_v = nn.Conv2d(in_channels=feat_dim,
                                       out_channels=25,
                                       kernel_size=final_cov_k,
                                       stride=1,
                                       padding=1 if final_cov_k == 3 else 0)

        self.predict_ann_index = nn.Conv2d(
            in_channels=feat_dim,
            out_channels=15,
            kernel_size=final_cov_k,
            stride=1,
            padding=1 if final_cov_k == 3 else 0)

        self.predict_uv_index = nn.Conv2d(in_channels=feat_dim,
                                          out_channels=25,
                                          kernel_size=final_cov_k,
                                          stride=1,
                                          padding=1 if final_cov_k == 3 else 0)

        self.inplanes = feat_dim

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return_dict = {}

        predict_uv_index = self.predict_uv_index(x)
        predict_ann_index = self.predict_ann_index(x)

        return_dict['predict_uv_index'] = predict_uv_index
        return_dict['predict_ann_index'] = predict_ann_index

        if self.with_uv:
            predict_u = self.predict_u(x)
            predict_v = self.predict_v(x)
            return_dict['predict_u'] = predict_u
            return_dict['predict_v'] = predict_v
        else:
            return_dict['predict_u'] = None
            return_dict['predict_v'] = None
            # return_dict['predict_u'] = torch.zeros(predict_uv_index.shape).to(predict_uv_index.device)
            # return_dict['predict_v'] = torch.zeros(predict_uv_index.shape).to(predict_uv_index.device)

        return return_dict


class SmplResNet(nn.Module):
    def __init__(self,
                 resnet_nums,
                 in_channels=3,
                 num_classes=229,
                 last_stride=2,
                 n_extra_feat=0,
                 truncate=0,
                 **kwargs):
        super().__init__()

        self.inplanes = 64
        self.truncate = truncate
        # extra = cfg.MODEL.EXTRA
        # self.deconv_with_bias = extra.DECONV_WITH_BIAS
        block, layers = resnet_spec[resnet_nums]

        self.conv1 = nn.Conv2d(in_channels,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=2) if truncate < 2 else None
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=last_stride) if truncate < 1 else None

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        if num_classes > 0:
            self.final_layer = nn.Linear(512 * block.expansion, num_classes)
            nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)

        self.n_extra_feat = n_extra_feat
        if n_extra_feat > 0:
            self.trans_conv = nn.Sequential(
                nn.Conv2d(n_extra_feat + 512 * block.expansion,
                          512 * block.expansion,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(512 * block.expansion, momentum=BN_MOMENTUM),
                nn.ReLU(True))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, infeat=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2) if self.truncate < 2 else x2
        x4 = self.layer4(x3) if self.truncate < 1 else x3

        if infeat is not None:
            x4 = self.trans_conv(torch.cat([infeat, x4], 1))

        if self.num_classes > 0:
            xp = self.avg_pooling(x4)
            cls = self.final_layer(xp.view(xp.size(0), -1))
            if not cfg.DANET.USE_MEAN_PARA:
                # for non-negative scale
                scale = F.relu(cls[:, 0]).unsqueeze(1)
                cls = torch.cat((scale, cls[:, 1:]), dim=1)
        else:
            cls = None

        return cls, {'x4': x4}

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                # state_dict = checkpoint
                state_dict_old = self.state_dict()
                for key in state_dict_old.keys():
                    if key in checkpoint.keys():
                        if state_dict_old[key].shape != checkpoint[key].shape:
                            del checkpoint[key]
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(
                        pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


class LimbResLayers(nn.Module):
    def __init__(self,
                 resnet_nums,
                 inplanes,
                 outplanes=None,
                 groups=1,
                 **kwargs):
        super().__init__()

        self.inplanes = inplanes
        block, layers = resnet_spec[resnet_nums]
        self.outplanes = 512 if outplanes == None else outplanes
        self.layer4 = self._make_layer(block,
                                       self.outplanes,
                                       layers[3],
                                       stride=2,
                                       groups=groups)

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes * groups,
                          planes * block.expansion * groups,
                          kernel_size=1,
                          stride=stride,
                          bias=False,
                          groups=groups),
                nn.BatchNorm2d(planes * block.expansion * groups,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, groups=groups))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.avg_pooling(x)

        return x
