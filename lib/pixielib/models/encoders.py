import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class ResnetEncoder(nn.Module):
    def __init__(self, append_layers = None):
        super(ResnetEncoder, self).__init__()
        from . import resnet
        # feature_size = 2048
        self.feature_dim = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        # regressor
        self.append_layers = append_layers
        # for normalize input images
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        self.register_buffer('MEAN', torch.tensor(MEAN)[None,:,None,None])
        self.register_buffer('STD', torch.tensor(STD)[None,:,None,None])
        
    def forward(self, inputs):
        ''' inputs: [bz, 3, h, w], range: [0,1]
        '''
        inputs = (inputs - self.MEAN)/self.STD
        features = self.encoder(inputs)
        if self.append_layers:
            features = self.last_op(features)
        return features

class MLP(nn.Module):
    def __init__(self, channels = [2048, 1024, 1], last_op = None):
        super(MLP, self).__init__()
        layers = []

        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l+1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        if last_op:
            layers.append(last_op)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outs = self.layers(inputs)
        return outs

class HRNEncoder(nn.Module):
    def __init__(self, append_layers = None):
        super(HRNEncoder, self).__init__()
        from . import hrnet
        self.feature_dim = 2048
        self.encoder = hrnet.load_HRNet(pretrained=True) #out: 2048
        # regressor
        self.append_layers = append_layers
        # for normalize input images
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        self.register_buffer('MEAN', torch.tensor(MEAN)[None,:,None,None])
        self.register_buffer('STD', torch.tensor(STD)[None,:,None,None])
        
    def forward(self, inputs):
        ''' inputs: [bz, 3, h, w], range: [0,1]
        '''
        inputs = (inputs - self.MEAN)/self.STD 
        features = self.encoder(inputs)['concat']
        if self.append_layers:
            features = self.last_op(features)
        return features   