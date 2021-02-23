from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from network import traj_critic, axis_aligned_iou   
import torchvision 
import cv2 
import pdb 
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
to_tensor = transforms.ToTensor()


got10k_dataset_path = "./data/GOT10k_train_val/"
result_path = "./benchmark/results/GOT10k_train_val/Tracker/"
attentionMap_path = "./temp_DIR_TO_SAVE_static_Global_attentionMap/"


# batchSize = 20 
lr = 1e-3    
num_epoch = 5000  
clip_len = 10 
img_size = 300   

traj_critic_net = traj_critic()
traj_critic_net = traj_critic_net.cuda() 

optimizer  = torch.optim.Adagrad(traj_critic_net.parameters(), lr=lr)
loss_fn    = torch.nn.L1Loss().cuda() 
videoFiles = os.listdir(attentionMap_path) 

traj_critic_net.train() 


#########################################################################################################
####                                         The Main Loop 
#########################################################################################################

for epochID in range(num_epoch): 

    epoch_totalLoss = 0

    for videoIndex in range(len(videoFiles)): 
        videoName = videoFiles[videoIndex] 
        
        result1_path = result_path + videoName + "/" + videoName + "_001.txt"
        result2_path = result_path + videoName + "/" + videoName + "_002.txt"
        gt_path      = result_path + videoName + "/" + videoName + "_groundtruth.txt"
        local_score_path  = result_path + videoName + "/" + videoName + "_scoreGlobal.txt" 
        global_score_path = result_path + videoName + "/" + videoName + "_scoreLocal.txt"
        
        #### tracking results and score. 
        tracking_result1 = np.loadtxt(result1_path, delimiter=',')  ## (90, 4) 
        tracking_result2 = np.loadtxt(result2_path, delimiter=',')  ## (90, 4) 
        gt               = np.loadtxt(gt_path, delimiter=',')       ## (90, 4) 
        local_score      = torch.from_numpy(np.loadtxt(local_score_path))             ## (90,) 
        global_score     = torch.from_numpy(np.loadtxt(global_score_path))            ## (90,) 

        local_score      = torch.unsqueeze(local_score, dim=1)
        global_score     = torch.unsqueeze(global_score, dim=1)
        
        #### image and attention maps. 
        imgFiles = os.listdir(got10k_dataset_path + videoName + "/")
        imgFiles = np.sort(imgFiles) 

        attentionFiles = os.listdir(attentionMap_path + videoName + "/") 
        attentionFiles = np.sort(attentionFiles) 

        init_imgPath = got10k_dataset_path + videoName + "/" + imgFiles[0]  
        init_image = cv2.imread(init_imgPath)
        init_bbox = gt[0] 

        # pdb.set_trace() 
        init_target = init_image[int(init_bbox[1]):int(init_bbox[1]+init_bbox[3]), int(init_bbox[0]):int(init_bbox[0]+init_bbox[2]),  :]
        init_target = cv2.resize(init_target, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('./init_target.png', init_target) 
        # cv2.imwrite('./init_image.png', init_image)

        startIndex = np.random.random_integers(len(attentionFiles) - clip_len) 

        image_list      = torch.zeros(clip_len, 3, img_size, img_size)
        initTarget_list = torch.zeros(clip_len, 3, img_size, img_size)
        attMap_list     = torch.zeros(clip_len, 3, img_size, img_size) 
        targetImg1_list = torch.zeros(clip_len, 3, img_size, img_size)
        targetImg2_list = torch.zeros(clip_len, 3, img_size, img_size)
        targetMap1_list = torch.zeros(clip_len, 3, img_size, img_size)
        targetMap2_list = torch.zeros(clip_len, 3, img_size, img_size)

        trajScore_list1 = torch.zeros(clip_len, 1)
        trajScore_list2 = torch.zeros(clip_len, 1)

        trajBBox_list1  = torch.zeros(clip_len, 4)
        trajBBox_list2  = torch.zeros(clip_len, 4)

        IoU_score_1 = 0
        IoU_score_2 = 0     

        #########################################################################################################
        ####                                            Load Batch data 
        #########################################################################################################
        count = 0 
        startIndex = 0 
        for INdex in range(startIndex, startIndex+clip_len):  
            imgPath = got10k_dataset_path + videoName + "/" + imgFiles[INdex]  
            image = cv2.imread(imgPath)
            
            imgIndex = int(imgFiles[INdex][:-4])  
            attMap_path = attentionMap_path + videoName + "/" + str(imgIndex)+"_dynamic_atttentonMAP_adaptIS.png" 
            attMap = cv2.imread(attMap_path)    ## (720, 1280, 3) 
            
            gt_curr        = gt[INdex] 
            result_curr1   = tracking_result1[INdex] 
            result_curr2   = tracking_result2[INdex] 

            #### BBox normalization 
            result_curr1[0] = max(0,  min(image.shape[1], result_curr1[0]))
            result_curr1[1] = max(0,  min(image.shape[0], result_curr1[1]))
            result_curr1[2] = max(10, min(image.shape[1], result_curr1[2]))
            result_curr1[3] = max(10, min(image.shape[0], result_curr1[3]))

            result_curr2[0] = max(0,  min(image.shape[1], result_curr2[0]))
            result_curr2[1] = max(0,  min(image.shape[0], result_curr2[1]))
            result_curr2[2] = max(10, min(image.shape[1], result_curr2[2]))
            result_curr2[3] = max(10, min(image.shape[0], result_curr2[3]))


            targetImg1     = image[int(result_curr1[1]):int(result_curr1[1]+result_curr1[3]), int(result_curr1[0]):int(result_curr1[0]+result_curr1[2]), :]  
            targetImg2     = image[int(result_curr2[1]):int(result_curr2[1]+result_curr2[3]), int(result_curr2[0]):int(result_curr2[0]+result_curr2[2]), :] 
            tagetattMap1   = attMap[int(result_curr1[1]):int(result_curr1[1]+result_curr1[3]), int(result_curr1[0]):int(result_curr1[0]+result_curr1[2]), :]  
            tagetattMap2   = attMap[int(result_curr2[1]):int(result_curr2[1]+result_curr2[3]), int(result_curr2[0]):int(result_curr2[0]+result_curr2[2]), :] 

            trajScore1 = local_score[INdex] 
            trajScore2 = global_score[INdex] 

            #### Normalization 
            image        = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC) 
            attMap       = cv2.resize(attMap, (img_size, img_size), interpolation=cv2.INTER_CUBIC) 
            targetImg1   = cv2.resize(targetImg1, (img_size, img_size), interpolation=cv2.INTER_CUBIC) 
            targetImg2   = cv2.resize(targetImg2, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            tagetattMap1 = cv2.resize(tagetattMap1, (img_size, img_size), interpolation=cv2.INTER_CUBIC) 
            tagetattMap2 = cv2.resize(tagetattMap2, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

            # cv2.imwrite('./image.png', image) 
            # cv2.imwrite('./attMap.png', attMap) 
            # cv2.imwrite('./targetImg1.png', targetImg1) 
            # cv2.imwrite('./targetImg2.png', targetImg2) 
            # cv2.imwrite('./tagetattMap1.png', tagetattMap1) 
            # cv2.imwrite('./tagetattMap2.png', tagetattMap2) 

            # pdb.set_trace() 

            image_list[count]      = to_tensor(image)  
            attMap_list[count]     = to_tensor(attMap)
            targetImg1_list[count] = to_tensor(targetImg1)
            targetImg2_list[count] = to_tensor(targetImg2)
            targetMap1_list[count] = to_tensor(tagetattMap1)
            targetMap2_list[count] = to_tensor(tagetattMap2)
            initTarget_list[count] = to_tensor(init_target)

            trajBBox_list1[count] = torch.from_numpy(result_curr1) 
            trajBBox_list2[count] = torch.from_numpy(result_curr2) 

            # pdb.set_trace()  

            trajScore_list1[count] = trajScore1
            trajScore_list2[count] = trajScore2   

            count = count + 1 
            

            #### Calculate the GIoU score 
            gt_curr[2] = gt_curr[0]+gt_curr[2];                         gt_curr[3] = gt_curr[1]+gt_curr[3]
            result_curr1[2] = result_curr1[0]+result_curr1[2];          result_curr1[3] = result_curr1[1]+result_curr1[3]
            result_curr2[2] = result_curr2[0]+result_curr2[2];          result_curr2[3] = result_curr2[1]+result_curr2[3]

            IoU_score_1 = IoU_score_1 + axis_aligned_iou(gt_curr, result_curr1) 
            IoU_score_2 = IoU_score_2 + axis_aligned_iou(gt_curr, result_curr2) 

            # pdb.set_trace() 
        
        optimizer.zero_grad() 
        pred_traj_score1 = traj_critic_net(image_list, attMap_list, targetImg1_list, targetMap1_list, initTarget_list, trajBBox_list1, trajScore_list1)
        pred_traj_score2 = traj_critic_net(image_list, attMap_list, targetImg2_list, targetMap2_list, initTarget_list, trajBBox_list2, trajScore_list2)

        IoU_score_1 = torch.from_numpy(np.array(IoU_score_1)).float().cuda() 
        IoU_score_2 = torch.from_numpy(np.array(IoU_score_2)).float().cuda() 
        traj_loss1 = loss_fn(pred_traj_score1, IoU_score_1)
        traj_loss2 = loss_fn(pred_traj_score2, IoU_score_2)
        total_loss = traj_loss1 + traj_loss2 


        # print('Epoch:', epochID, "     video: ", videoName, "     loss:", total_loss.item())

        # backward + optimize
        total_loss.backward()
        optimizer.step()


        epoch_totalLoss = epoch_totalLoss + total_loss.item()
    # Save weights
    if epochID % 50 == 0:
        print('==>> Epoch:', epochID, ' ==>> Train_loss->', epoch_totalLoss) 
        checkpointName = str(epochID) + "_traj_critic_net.pkl" 
        torch.save(traj_critic_net.state_dict(), './traj_measure_model_checkoints/'+checkpointName)













































































































































