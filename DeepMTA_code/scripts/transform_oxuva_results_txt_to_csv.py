import os 
import numpy as np 
import cv2
import time
import oxuva
import pdb 
from skimage import measure
import json
import pdb
import cv2
import os
import pandas as pd
resultpath= '/home/wangxiao/Documents/deepMTA_project/DeepMTA_TCSVT_project/benchmark/results/oxuva_txt_files/' 
videopath="/home/wangxiao/dataset/OxUvA/images/test/"
videos=os.listdir(videopath)
txtFiles = os.listdir(resultpath) 

attMap_path = "/home/wangxiao/Documents/deepMTA_project/DeepMTA_TCSVT_project/temp_DIR_TO_SAVE_static_Global_attentionMap/"

# export PYTHONPATH="/home/wangxiao/Documents/deepMTA_project/DeepMTA_TCSVT_project/long-term-tracking-benchmark-master/python:$PYTHONPATH"

for i in range(len(videos)):
    txtName = videos[i] + ".txt"
    preds = np.loadtxt(resultpath + txtName, delimiter=',') 

    print("==>> txtName: ", txtName) 
    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    video_ids=[]
    obj_ids=[]
    frame_nums=[]
    presents=[]
    scores=[]
    video_id=videos[i][0:7]
    if(len(videos[i])==7):
        obj_id='obj0000'
    elif(videos[i][-1]=='2'):
        obj_id='obj0001'
    else:
        obj_id='obj0002'
    
    score = 0.5
    # l=result['res']

    imgs = os.listdir(videopath+videos[i]+'/')
    imgs = np.sort(imgs) 
    # pdb.set_trace() 

    image = cv2.imread(videopath+videos[i]+'/'+imgs[0])
    imgh  = image.shape[0]
    imgw  = image.shape[1]

    attvideo_attPath = attMap_path + videos[i] + "/"
    attFiles = os.listdir(attvideo_attPath) 


    occurFlag_list = [] 

    if len(attFiles)+1 == len(imgs): 
        ###############################################################################
        #### 					Scan the Attention Map 
        ###############################################################################
        occurFlag_list.append(1)
        for j in range(len(imgs)-1):
            attMap = cv2.imread(attvideo_attPath + attFiles[j]) 
            ret, static_atttentonMAP = cv2.threshold(attMap, 5, 255, cv2.THRESH_BINARY)
            label_image = measure.label(static_atttentonMAP)
            props = measure.regionprops(label_image)

            if len(props) > 0: 
                occurFlag_list.append(1) 
            else: 
                occurFlag_list.append(0) 
    else: 
        for j in range(len(imgs)):
            occurFlag_list.append(1) 

    # pdb.set_trace() 

    for j in range(len(imgs)):

        x=preds[j][0]
        y=preds[j][1]
        w=preds[j][2]
        h=preds[j][3]

        ## results relative to original image size. 
        x1=x/imgw
        x2=(x+w)/imgw
        y1=y/imgh
        y2=(y+h)/imgh
 

        if j >= 5 and  np.sum(occurFlag_list[j-5:j]) == 0: 
            present = 'False' 
            print("==>> got one missing ......")
        else: 
            present = 'True' 

        x1=round(x1,4)
        x2=round(x2,4)
        y1=round(y1,4)
        y2=round(y2,4)

        frame=imgs[j][0:6]

        if(frame=='000000'):
            frame_num=0
        else:
            frame_num=frame.lstrip('0')

        xmin.append(x1)
        xmax.append(x2)
        ymin.append(y1)
        ymax.append(y2)
        video_ids.append(video_id)
        obj_ids.append(obj_id)
        frame_nums.append(frame_num)
        presents.append(present)
        scores.append(score)

    # pdb.set_trace() 

    dataframe=pd.DataFrame({'video_id':video_ids,'object_id':obj_ids,'frame_num':frame_nums,'present':presents,\
                            'score':scores,'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax})
    savepath='./oxuva_csv_results_missFlag/' +videos[i][0:7]+'_'+obj_id+'.csv'
    columns=['video_id','object_id','frame_num','present','score','xmin','xmax','ymin','ymax']

    dataframe.to_csv(savepath,index=False,columns=columns,header=None)


    # pdb.set_trace() 
