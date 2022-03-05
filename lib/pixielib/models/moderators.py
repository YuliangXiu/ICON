''' Moderator
# Input feature: body, part(head, hand)
# output: fused feature, weight
'''
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

## MLP + temperature softmax
## w = SoftMax(w^\prime * temperature)
class TempSoftmaxFusion(nn.Module):
    def __init__(self, channels = [2048*2, 1024, 1], detach_inputs=False, detach_feature=False):
        super(TempSoftmaxFusion, self).__init__()
        self.detach_inputs = detach_inputs
        self.detach_feature = detach_feature
        # weight
        layers = []
        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l+1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        # temperature
        self.register_parameter('temperature', nn.Parameter(torch.ones(1)))

    def forward(self, x, y, work=True):
        '''
        x: feature from body
        y: feature from part(head/hand) 
        work: whether to fuse features
        '''
        if work:
            # 1. cat input feature, predict the weights
            f_in = torch.cat([x, y], dim=1)
            if self.detach_inputs:
                f_in = f_in.detach()
            f_temp = self.layers(f_in)
            f_weight = F.softmax(f_temp*self.temperature, dim=1)

            # 2. feature fusion
            if self.detach_feature:
                x = x.detach()
                y = y.detach()
            f_out = f_weight[:,[0]]*x + f_weight[:,[1]]*y
            x_out = f_out
            y_out = f_out
        else:
            x_out = x
            y_out = y
            f_weight = None
        return x_out, y_out, f_weight

## MLP + Gumbel-Softmax trick
## w = w^{\prime} - w^{\prime}\text{.detach()} + w^{\prime}\text{.gt(0.5)}
class GumbelSoftmaxFusion(nn.Module):
    def __init__(self, channels = [2048*2, 1024, 1], detach_inputs=False, detach_feature=False):
        super(GumbelSoftmaxFusion, self).__init__()
        self.detach_inputs = detach_inputs
        self.detach_feature = detach_feature

        # weight
        layers = []
        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l+1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Softmax())
        self.layers = nn.Sequential(*layers)

    def forward(self, x, y, work=True):
        '''
        x: feature from body
        y: feature from part(head/hand) 
        work: whether to fuse features
        '''
        if work:
            # 1. cat input feature, predict the weights
            f_in = torch.cat([x, y], dim=-1)
            if self.detach_inputs:
                f_in = f_in.detach()
            f_weight = self.layers(f_in)
            # weight to be hard
            f_weight = f_weight - f_weight.detach() + f_weight.gt(0.5)
            
            # 2. feature fusion
            if self.detach_feature:
                x = x.detach()
                y = y.detach()
            f_out = f_weight[:,[0]]*x + f_weight[:,[1]]*y
            x_out = f_out
            y_out = f_out
        else:
            x_out = x
            y_out = y
            f_weight = None
        return x_out, y_out, f_weight


