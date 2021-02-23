import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models

import torchvision.ops as torchops

import math
from torch.autograd import Variable
import pdb 

from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

import numpy as np
import cv2 
import pdb 



class traj_critic(nn.Module):
    def __init__(self):
        super(traj_critic, self).__init__()
        #### ResNet model 
        caffenet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(caffenet.children())[:-1])

        self.trajBBox_linear = nn.Linear(4, 32) 
        self.trajScore_linear = nn.Linear(10, 32) 
        self.imgReducDIM_linear = nn.Linear(2560, 512) 

        self.regressor = nn.Sequential(
                nn.Linear(5472, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, 1),
                )
        self.relu = nn.ReLU(inplace=True)




    def forward(self, img, attMap, targetImg, targetAtten, initTarget, trajBBox, trajScore):       
        img_feat        = self.encoder(img.cuda())      ## torch.Size([10, 512, 1, 1]) 
        img_feat        = self.relu(img_feat)

        attMap_feat     = self.encoder(attMap.cuda())      
        attMap_feat     = self.relu(attMap_feat)

        targetImg_feat  = self.encoder(targetImg.cuda())      
        targetImg_feat  = self.relu(targetImg_feat)

        targetAtt_feat  = self.encoder(targetAtten.cuda())      
        targetAtt_feat  = self.relu(targetAtt_feat)

        initTarget_feat = self.encoder(initTarget.cuda())   
        initTarget_feat = self.relu(initTarget_feat)



        fused1 = torch.cat((img_feat, attMap_feat), 1)              ## torch.Size([10, 1024, 1, 1]) 
        fused2 = torch.cat((targetImg_feat, targetAtt_feat), 1)     ## torch.Size([10, 1024, 1, 1])  
        fused2 = torch.cat((fused2, initTarget_feat), 1) 
        fused3 = torch.cat((fused1, fused2), 1)     ## torch.Size([10, 2560, 1, 1]) 

        fused3 = torch.squeeze(fused3, dim=2)
        fused3 = torch.squeeze(fused3, dim=2)        
        fused3 = self.imgReducDIM_linear(fused3)
        fused3 = fused3.view(-1)

        trajBBox_feat  = self.trajBBox_linear(trajBBox.cuda()) 
        trajBBox_feat  = trajBBox_feat.view(-1)
        trajBBox_feat  = self.relu(trajBBox_feat)

        trajScore      = torch.transpose(trajScore, 0, 1) 
        trajScore_feat = self.trajScore_linear(trajScore.cuda()) 
        trajScore_feat = trajScore_feat.view(-1)
        trajScore_feat  = self.relu(trajScore_feat)

        fused4 = torch.cat((trajBBox_feat, trajScore_feat)) ## 352-D 
        final_feat = torch.cat((fused3, fused4)) ## 5472-D 
        final_feat = self.relu(final_feat) 
        pred_traj_score = self.regressor(final_feat)
 
        return pred_traj_score 



































def axis_aligned_iou(boxA, boxB):
    # make sure that x1,y1,x2,y2 of a box are valid
    assert(boxA[0] <= boxA[2])
    assert(boxA[1] <= boxA[3])
    assert(boxB[0] <= boxB[2])
    assert(boxB[1] <= boxB[3])

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou







































































































