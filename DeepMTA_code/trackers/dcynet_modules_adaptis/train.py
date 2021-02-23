from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from torch.autograd import Variable

from data_loader import DataLoader
from generator import DC_adaIS_Generator
from utils import *
import pdb
import warnings
warnings.filterwarnings("ignore")


batch_size = 10 
lr = 1e-4


generator = DC_adaIS_Generator()

#### load pre-trained model 
# print("==>> Loading pre-trained model ... ")
# generator.load_state_dict(torch.load('./dynamic_global_search_region_generator.pkl'))
# # generator = torch.load('./dynamic_global_search_region_generator.pkl')
# print("==>> Done !!!")

if torch.cuda.is_available():
    generator.cuda()

criterion = nn.BCELoss()


print("===================================================================================")
print("===================================================================================")
print(generator)
print("===================================================================================")
print("===================================================================================")

g_optim = torch.optim.Adagrad(generator.parameters(), lr=lr)

num_epoch = 50 
dataloader = DataLoader(batch_size)
num_batch = 500  
print("==>> num_batch: ", num_batch)


def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

counter = 0
start_time = time.time()
DIR_TO_SAVE = "./generator_output/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)



print("###################################################################################")
print("                             The Main Training Loop                                ")
print("###################################################################################")

generator.train()

for current_epoch in range(num_epoch):
    n_updates = 1
    d_cost_avg = 0
    g_cost_avg = 0
    
    for idx in range(int(num_batch)):
        (batch_img, batch_map, targetObject_img, coords) = dataloader.get_batch()
        batch_img = to_variable(batch_img, requires_grad=True)
        batch_map = to_variable(batch_map, requires_grad=False)
        targetObject_img = to_variable(targetObject_img, requires_grad=True)
        # batch_map = nn.functional.interpolate(batch_map, size=[216, 216])
        
        val_batchImg = batch_img
        val_targetObjectImg = targetObject_img
        val_coords = coords 
        
        g_optim.zero_grad()
        attention_map = generator(batch_img, targetObject_img, coords)
        
        # pdb.set_trace()
        g_gen_loss = criterion(attention_map, batch_map)
        g_loss = torch.sum(g_gen_loss)
        g_cost_avg += g_loss.item()
        g_loss.backward()
        g_optim.step()

        n_updates += 1

        if (idx+1)%100 == 0:
            print("==>> Epoch [%d/%d], Step[%d/%d], g_gen_loss: %.4f, LR: %.6f, time: %4.4f" % \
                (current_epoch, num_epoch, idx+1, num_batch, g_loss.item(), lr, time.time()-start_time))
        counter += 1 

    # pdb.set_trace()
    g_cost_avg /= num_batch

    # Save weights every 3 epoch
    if current_epoch % 3 == 0:
        print('==>> Epoch:', current_epoch, ' ==>> Train_loss->', (g_cost_avg))
        torch.save(generator.state_dict(), 'generator_dcyNet_adaIS_1e4.pkl')

    # validation 
    out = generator(val_batchImg, val_targetObjectImg, val_coords)
    map_out = out.cpu().data.squeeze(0)
    for iiidex in range(5): 
       new_path = DIR_TO_SAVE + str(current_epoch) + str(iiidex) + ".jpg"
       pilTrans = transforms.ToPILImage()
       pilImg = pilTrans(map_out[iiidex]) 
       print('==>> Image saved to ', new_path)
       pilImg.save(new_path)



