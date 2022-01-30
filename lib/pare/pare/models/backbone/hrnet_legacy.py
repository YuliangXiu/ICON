import timm
from torch import nn

models = [
    'hrnet_w18_small',
    'hrnet_w18_small_v2',
    'hrnet_w18',
    'hrnet_w30',
    'hrnet_w32',
    'hrnet_w40',
    'hrnet_w44',
    'hrnet_w48',
    'hrnet_w64',
]


class HRNet(nn.Module):
    def __init__(self, arch, pretrained=True):
        super(HRNet, self).__init__()
        self.m = timm.create_model(arch, pretrained=pretrained)

    def forward(self, x):
        return self.m.forward_features(x)


def hrnet_w32(pretrained=True):
    return HRNet('hrnet_w32', pretrained)


def hrnet_w48(pretrained=True):
    return HRNet('hrnet_w48', pretrained)


def hrnet_w64(pretrained=True):
    return HRNet('hrnet_w64', pretrained)


def dla34(pretrained=True):
    return HRNet('dla34', pretrained)
