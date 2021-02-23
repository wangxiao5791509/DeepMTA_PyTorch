# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join, realpath, dirname, exists, isdir
from os import listdir
import logging
import glob
import numpy as np
import json
from collections import OrderedDict
import functools

import pdb 





def get_dataset_zoo():
    root = realpath(join(dirname(__file__), '../../data'))
    zoos = listdir(root)

    def valid(x):
        y = join(root, x)
        if not isdir(y): return False

        return exists(join(y, 'list.txt')) \
               or exists(join(y, 'train', 'meta.json'))\
               or exists(join(y, 'ImageSets', '2016', 'val.txt'))

    zoos = list(filter(valid, zoos))
    return zoos


dataset_zoo = get_dataset_zoo()

def load_tasks_with_annotations(fname):
    with open(fname, 'r') as fp:
        if fname.endswith('.csv'):
            tracks = oxuva.load_dataset_annotations_csv(fp)
        else:
            raise ValueError(f"unknown extension: {fname}")
    return oxuva.map_dict(oxuva.make_task_from_track, tracks)






def load_dataset(dataset):

    ##################################################################
    ####    VOT2018, VOT2018-LT, OTB2015, GOT10k, LaSOT, OxUVA  
    ##################################################################    

    info = OrderedDict()
    if 'VOT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            logging.error("Please download test dataset!!!")
            exit()
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            if gt.shape[1] == 4:
                gt = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3]-1,
                                      gt[:, 0] + gt[:, 2]-1, gt[:, 1] + gt[:, 3]-1, gt[:, 0] + gt[:, 2]-1, gt[:, 1]))
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}


    elif 'VOT2018-LT' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            logging.error("Please download test dataset!!!")
            exit()
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            if gt.shape[1] == 4:
                gt = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3]-1,
                                      gt[:, 0] + gt[:, 2]-1, gt[:, 1] + gt[:, 3]-1, gt[:, 0] + gt[:, 2]-1, gt[:, 1]))
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}


    elif 'OTB' in dataset:
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            print("Please download OTB dataset into data folder")
        json_path = base_path + '.json'
        info = json.load(open(json_path, 'r'))
      
        # load the video frames
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
            info[v]['gt'] = np.array(info[v]['gt_rect'])-[1,1,0,0]  # our tracker is 0-index
            info[v]['name'] = v


    elif 'GOT' in dataset: 
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            print("Please download GOT10K dataset into data folder")

        json_path = base_path + '.json' 
        info = json.load(open(json_path, 'r'))


    elif 'GOT10k_train_val' in dataset: 
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            print("Please download GOT10k_train_val dataset into data folder")

        json_path = base_path + '.json' 
        info = json.load(open(json_path, 'r'))



    elif 'LaSOT' in dataset: 
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            print("Please download LaSOT dataset into data folder")
        json_path = base_path + '.json' 
        info = json.load(open(json_path, 'r'))

    elif 'UAV20L' in dataset: 
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            print("Please download UAV20L dataset into data folder")
        json_path = base_path + '.json' 
        info = json.load(open(json_path, 'r'))

    elif 'OXUVA' in dataset: 
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            print("Please download OXUVA dataset into data folder")
        json_path = base_path + '.json' 
        info = json.load(open(json_path, 'r'))

    elif 'TC128' in dataset: 
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            print("Please download TC128 dataset into data folder")
        json_path = base_path + '.json' 
        info = json.load(open(json_path, 'r'))        

    elif 'UAV123' in dataset: 
        base_path = join(realpath(dirname(__file__)), '../../data', dataset)
        if not exists(base_path):
            print("Please download UAV123 dataset into data folder")
        json_path = base_path + '.json' 
        info = json.load(open(json_path, 'r'))


    else:
        logging.error(f'{dataset} not supported')
        exit()
    return info
