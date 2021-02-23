import glob
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from constants import *
import pdb 
import random

class DataLoader(object):

    def __init__(self, batch_size = 5):
        #reading data list
        # self.list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToResizedImagesTrain, '*train*'))]
        self.list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToResizedImagesTrain, '*image*'))]
        self.batch_size = batch_size
        self.size = len(self.list_img)
        self.cursor = 0
        self.num_batches = self.size / batch_size

    def get_batch(self): # Returns 
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)
        
        mask_size = 216  
        img = torch.zeros(self.batch_size, 3, 300, 300)
        sal_map = torch.zeros(self.batch_size, 1, mask_size, mask_size)
        targetObject = torch.zeros(self.batch_size, 3, 100, 100)
        coords = torch.zeros(self.batch_size, 2) 
        
        #to_tensor = transform.Compose(transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
        to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.

        for idx in range(self.batch_size):
            curr_file = self.list_img[self.cursor]

            # pdb.set_trace()
            temp_index2 = curr_file.find('_')
            videoName = curr_file[:temp_index2]

            imgIndex = curr_file[temp_index2+7:]

            # pdb.set_trace()
            
            targetObject_img_path = os.path.join(pathToResizedTargetObjectTrain, videoName + '_targetObject.jpg')
            full_img_path = os.path.join(pathToResizedImagesTrain, videoName + "_image-" + imgIndex + '.jpg')
            full_map_path = os.path.join(pathToResizedMapsTrain, videoName + "_mask-" + imgIndex + '.jpg')
            self.cursor += 1
            inputimage = cv2.imread(full_img_path) # (192,256,3)


            # pdb.set_trace()
            img[idx] = to_tensor(inputimage)
            
            targetObjectimage = cv2.imread(targetObject_img_path)
            targetObject[idx] = to_tensor(targetObjectimage)
            
            saliencyimage = cv2.imread(full_map_path, 0)
            saliencyimage = cv2.resize(saliencyimage, (mask_size, mask_size), interpolation=cv2.INTER_CUBIC)

            num_points = 1
            possible_points = np.where(saliencyimage)
            num_possible_points = possible_points[0].shape[0]
            rindx = random.sample(list(range(num_possible_points)), k=min(num_points, num_possible_points))
            points = []
            for j in rindx:
                points.append((possible_points[0][j], possible_points[1][j]))
            points = np.array(points)
            coords[idx] = torch.from_numpy(points)  


            saliencyimage = np.expand_dims(saliencyimage, axis=2)
            sal_map[idx] = to_tensor(saliencyimage)

            
            
        return (img, sal_map, targetObject, coords)

        
