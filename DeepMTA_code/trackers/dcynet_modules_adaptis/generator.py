import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torchvision.ops as torchops

import math
from torch.autograd import Variable
from ops import * 
import pdb 

from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

from resnet import resnet18 
import numpy as np
import cv2 
import pdb 


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [maxpool2d()]
        else:
            conv = conv2d(in_channels, v)
            layers += [conv, relu(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_deconv_layers(cfg):
    layers = []
    in_channels = 512*2
    for v in cfg:
        if v == 'U':
            layers += [nn.Upsample(scale_factor=2)]
        else:
            deconv = deconv2d(in_channels, v)
            layers += [deconv]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
}

class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class AdaptiveConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(AdaptiveConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)

    def forward(self, input, dynamic_weight):
        # Get batch num
        batch_num = input.size(0)

        # Reshape input tensor from size (N, C, H, W) to (1, N*C, H, W)
        input = input.view(1, -1, input.size(2), input.size(3))

        # Reshape dynamic_weight tensor from size (N, C, H, W) to (1, N*C, H, W)
        dynamic_weight = dynamic_weight.view(-1, 1, dynamic_weight.size(2), dynamic_weight.size(3))

        # Do convolution
        conv_rlt = F.conv2d(input, dynamic_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Reshape conv_rlt tensor from (1, N*C, H, W) to (N, C, H, W)
        conv_rlt = conv_rlt.view(batch_num, -1, conv_rlt.size(2), conv_rlt.size(3))

        return conv_rlt


def encoder():
    return make_conv_layers(cfg['E'])

def decoder():
    return make_deconv_layers(cfg['D'])


class AddCoords(nn.Module):

    def __init__(self, ):
        super().__init__() 

    def forward(self, input_tensor, points):
        _, x_dim, y_dim = input_tensor.size()
        batch_size = 1 

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)    ## torch.Size([1, 9, 9]) 
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)    ## torch.Size([1, 9, 9]) 

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        coords = torch.cat((xx_channel, yy_channel), dim=1)     ## torch.Size([20, 2, 9, 9])
        coords = coords.type(torch.FloatTensor)

        add_xy = torch.reshape(points, (1, 2, 1))   ## torch.Size([1, 2, 1]) 
        add_xy_ = add_xy.repeat(1, 1, x_dim * y_dim)  ## torch.Size([1, 2, 81])
        add_xy_ = torch.reshape(add_xy_, (1, 2, x_dim, y_dim))  ## torch.Size([1, 2, 9, 9]) 
        add_xy_ = add_xy_.type(torch.FloatTensor)

        coords = (coords - add_xy_)     ## torch.Size([1, 2, 9, 9]) 
        coord_features = np.clip(np.array(coords), -1, 1)   ## (1, 2, 9, 9) 
        coord_features = torch.from_numpy(coord_features).cuda() 

        return coord_features



class FCController(nn.Module):
    def __init__(self):
        super(FCController, self).__init__()

        self.fc_net = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, 512),
                )
        
    def forward(self, input_feat):
        adaIN_input = self.fc_net(input_feat) 

        return adaIN_input


def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std




class AdaIN_layer(nn.Module):
    def __init__(self, channels = 512, norm=True):
        super(AdaIN_layer, self).__init__()

        self.channels = channels
        self.norm = norm
        self.affine_scale = nn.Linear(channels, channels) 
        self.affine_bias = nn.Linear(channels, channels) 



    def forward(self, x, control_feats):
        # control_feats: torch.Size([20, 512]) 
        # x: torch.Size([20, 512, 18, 18]) 
        ys = self.affine_scale(control_feats) ## torch.Size([20, 512])
        yb = self.affine_bias(control_feats)  ## torch.Size([20, 512])

        ys = torch.unsqueeze(ys, dim=2)  ## torch.Size([20, 512, 1]) 
        yb = torch.unsqueeze(yb, dim=2)  ## torch.Size([20, 512, 1]) 

        # style_mean, style_std = calc_mean_std(control_feats)

        # xm.shape: torch.Size([20, 512, 324])
        xm = x.view(x.shape[0], 512, -1)    # (N,C,H,W) --> (N,C,K)  

        # import pdb 
        # pdb.set_trace() 

        if self.norm:
            xm_mean = torch.mean(xm, dim=2)  ## torch.Size([20, 512, 1])
            xm_mean = torch.unsqueeze(xm_mean, dim=2) 
            xm_centered = xm - xm_mean    ## torch.Size([20, 512, 324])
            xm_temp = torch.mean(torch.pow(xm_centered, 2), dim=2) 
            xm_temp = torch.unsqueeze(xm_temp, dim=2) 
            xm_std_rev = torch.rsqrt(xm_temp)  ## torch.Size([20, 512, 1]) 
            xm_norm = xm_centered / xm_std_rev
        else:
            xm_norm = xm

        xm_scaled = (xm_norm * ys) + yb 
        xm_scaled = xm_scaled.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        return xm_scaled








class DC_adaIS_Generator(nn.Module):
    def __init__(self):
        super(DC_adaIS_Generator, self).__init__()
        self.encoder = resnet18() 
        self.decoder = decoder()
        self.mymodules = nn.ModuleList([deconv2d(64, 1, kernel_size=1, padding = 0), nn.Sigmoid()]) 
        num_of_feat = 512
        self.relu = nn.ReLU(inplace=True)

        self.addcoords = AddCoords()
        self.AdaIN = AdaIN_layer() 
        self.fc_controler = FCController()


        ##################################################################################
        ##                              Decoder part 
        ##################################################################################        
        self.CT_1 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.CT_2 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.CT_3 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.Upsamp_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.CT_4 = nn.ConvTranspose2d(514, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.CT_5 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.CT_6 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.Upsamp_2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.CT_7 = nn.ConvTranspose2d(512*2, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.CT_8 = nn.ConvTranspose2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.CT_9 = nn.ConvTranspose2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.Upsamp_3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.CT_10 = nn.ConvTranspose2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.CT_11 = nn.ConvTranspose2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.Upsamp_4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.CT_12 = nn.ConvTranspose2d(640, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.CT_13 = nn.ConvTranspose2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.Upsamp_5 = nn.Upsample(scale_factor=3, mode="nearest")

        self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)






    def forward(self, x, targetObject_img, coords):      

        x2_feat, x3_feat, x4_feat  = self.encoder(x)  
        ## (torch.Size([20, 128, 38, 38]), torch.Size([20, 256, 19, 19]), torch.Size([20, 512, 10, 10])) 

        targetObject_img = nn.functional.interpolate(targetObject_img, size=[100, 100])
        con_x2_feat, con_x3_feat, con_x4_feat = self.encoder(targetObject_img)    
        ## (torch.Size([20, 128, 13, 13]), torch.Size([20, 256, 7, 7]), torch.Size([20, 512, 4, 4]))

        # Mutual Adaptation Module 
        DC_2 = AdaptiveConv2d(x2_feat.size(0) * x2_feat.size(1),  x2_feat.size(0) * x2_feat.size(1), 5, padding=1, \
                                groups=x2_feat.size(0) * x2_feat.size(1), bias=False)
        DC_3 = AdaptiveConv2d(x3_feat.size(0) * x3_feat.size(1),  x3_feat.size(0) * x3_feat.size(1), 5, padding=1, \
                                groups=x3_feat.size(0) * x3_feat.size(1), bias=False)
        DC_4 = AdaptiveConv2d(x4_feat.size(0) * x4_feat.size(1),  x4_feat.size(0) * x4_feat.size(1), 5, padding=1, \
                                groups=x4_feat.size(0) * x4_feat.size(1), bias=False)

        dc_feats_2 = DC_2(x2_feat, con_x2_feat)     ## torch.Size([20, 128, 28, 28]) 
        # dc_feats_2 = self.relu(dc_feats_2)

        dc_feats_3 = DC_3(x3_feat, con_x3_feat)     ## torch.Size([20, 256, 15, 15])
        # dc_feats_3 = self.relu(dc_feats_3)

        dc_feats_4 = DC_4(x4_feat, con_x4_feat)     ## torch.Size([20, 512, 9, 9])
        # dc_feats_4 = self.relu(dc_feats_4)


        gated_2 = torch.sigmoid(dc_feats_2) 
        gated_3 = torch.sigmoid(dc_feats_3) 
        gated_4 = torch.sigmoid(dc_feats_4) 

        gated_output_2 = gated_2 * dc_feats_2   ## torch.Size([20, 128, 28, 28])
        gated_output_3 = gated_3 * dc_feats_3   ## torch.Size([20, 256, 15, 15])
        gated_output_4 = gated_4 * dc_feats_4   ## torch.Size([20, 512, 9, 9]) 

        # encoded_feat = gated_output_2 + gated_output_3 + gated_output_4

        # pdb.set_trace() 
        gated_output_3 = nn.functional.interpolate(gated_output_3, size=[18, 18])   ## torch.Size([20, 256, 18, 18])
        gated_output_2 = nn.functional.interpolate(gated_output_2, size=[36, 36]) 

        ####################################################
        ######            decoding + concat path
        ####################################################
        gated_output_4 = self.CT_1(gated_output_4)
        gated_output_4 = self.CT_2(gated_output_4)
        gated_output_4 = self.CT_3(gated_output_4)
        gated_output_4 = self.relu(gated_output_4)
        ## gated_output_4.shape: torch.Size([20, 512, 9, 9])

        gated_output_4_new = torch.zeros(gated_output_4.shape[0], gated_output_4.shape[1]+2, gated_output_4.shape[2], gated_output_4.shape[3])

        # pdb.set_trace() 

        for point_idx in range(gated_output_4.shape[0]): 
            feat_map = gated_output_4[point_idx] 
            point = coords[point_idx] 

            coords_feat = self.addcoords(feat_map, point)
            coords_feat = torch.squeeze(coords_feat, dim=0)
            fused_feats = torch.cat((coords_feat, feat_map), dim=0)
            gated_output_4_new[point_idx] = fused_feats 

        gated_output_4_new = gated_output_4_new.cuda() 



        bi = torch.arange(coords.shape[0])
        bi = torch.unsqueeze(bi, dim=1)     ## (batchSize, 1)
        rois = torch.cat((coords * 9 // 300, (coords) * 9// 300), dim=1)   ## (x1, y1, x2, y2) 
        bi = bi.type(torch.FloatTensor)
        rois = torch.cat((bi, rois), dim=1).cuda()  
        output_size = (1, 1)

        www = torchops.roi_pool(gated_output_4, rois, output_size, spatial_scale=1.0)    ## torch.Size([20, 512, 1, 1]) 
        www = torch.squeeze(www, dim=2)     ## (20, 512, 1)
        www = torch.squeeze(www, dim=2)     ## (20, 512)

        adaIN_input = self.fc_controler(www)     ## (20, 512) 




        # pdb.set_trace() 

        dc_feats_4 = self.CT_4(gated_output_4_new) 
        dc_feats_4 = self.CT_5(dc_feats_4)
        dc_feats_4 = self.CT_6(dc_feats_4)
        dc_feats_4 = self.relu(dc_feats_4)
        up_d4 = self.Upsamp_2(dc_feats_4)       ##  dc_feats_4: torch.Size([20, 512, 9, 9])         
                                                ##  up_d4       torch.Size([20, 512, 18, 18]) 

        AdaIN_output = self.AdaIN(up_d4, adaIN_input)   ## torch.Size([20, 512, 18, 18]) 
        AdaIN_output = torch.cat((AdaIN_output, up_d4), dim=1)

        # pdb.set_trace() 
        dc_feats_3 = self.CT_7(AdaIN_output)
        dc_feats_3 = self.CT_8(torch.cat((dc_feats_3, gated_output_3), dim=1))
        dc_feats_3 = self.CT_9(dc_feats_3)        
        dc_feats_3 = self.relu(dc_feats_3) 
        up_d3 = self.Upsamp_3(dc_feats_3)       ##  up_d3: torch.Size([20, 768, 36, 36])     
                                                     
        dc_feats_2 = self.CT_10(up_d3)          ##  dc_feats_2: torch.Size([20, 512, 36, 36])
        dc_feats_2 = self.CT_11(torch.cat((dc_feats_2, gated_output_2), dim=1))
        dc_feats_2 = self.relu(dc_feats_2)      ##  dc_feats_2: torch.Size([20, 640, 36, 36]) 


        up_d2 = self.Upsamp_4(dc_feats_2)       ## torch.Size([20, 640, 72, 72])

        dc_feats_1 = self.CT_12(up_d2)
        dc_feats_1 = self.CT_13(dc_feats_1)
        dc_feats_1 = self.relu(dc_feats_1)       


        dc_feats_1 = self.Upsamp_5(dc_feats_1)       
        dc_feats_1 = self.relu(dc_feats_1)      ## torch.Size([20, 64, 216, 216]) 

        # pdb.set_trace() 

        # output = self.Conv_1x1(dc_feats_1)
        output = self.mymodules[0](dc_feats_1)
        output = self.mymodules[1](output)

        # output = nn.functional.interpolate(output, size=[300, 300])

        return output 


