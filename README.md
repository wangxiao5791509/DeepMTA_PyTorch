# DeepMTA_PyTorch

### Officical PyTorch Implementation of "Dynamic Attention-guided Multi-TrajectoryAnalysis for Single Object Tracking", Xiao Wang, Zhe Chen, Jin Tang, Bin Luo, Yaowei Wang, Yonghong Tian, Feng Wu, IEEE Transactions on Circuits and Systems for Video Technology (T-CSVT 2021) [[Paper](https://ieeexplore.ieee.org/document/9345930)] [[Project](https://sites.google.com/view/mt-track/home)] 


## Abstract: 
Most of the existing single object trackers track the target in a unitary local search window, making them particularly vulnerable to challenging factors such as heavy occlusions and out-of-view movements. Despite the attempts to further incorporate global search, prevailing mechanisms that cooperate local and global search are relatively static, thus are still sub-optimal for improving tracking performance. By further studying the local and global search results, we raise a question: can we allow more dynamics for cooperating both results? In this paper, we propose to introduce more dynamics by devising a dynamic attention-guided multi-trajectory tracking strategy. In particular, we construct dynamic appearance model that contains multiple target templates, each of which provides its own attention for locating the target in the new frame. Guided by different attention, we maintain diversified tracking results for the target to build multi-trajectory tracking history, allowing more candidates to represent the true target trajectory. After spanning the whole sequence, we introduce a multi-trajectory selection network to find the best trajectory that deliver improved tracking performance. Extensive experimental results show that our proposed tracking strategy achieves compelling performance on various large-scale tracking benchmarks.


## Our Proposed Approach: 
![fig-1](https://github.com/wangxiao5791509/DeepMTA_PyTorch/blob/master/figures/pipeline.png)




## Install: 
~~~
git clone https://github.com/wangxiao5791509/DeepMTA_PyTorch
cd DeepMTA_TCSVT_project

# create the conda environment
conda env create -f environment.yml
conda activate deepmta

# build the vot toolkits
bash benchmark/make_toolkits.sh
~~~

## Download Dataset and Model: 
download pre-trained **Traj-Evaluation-Network** [[Onedrive](https://stuahueducn-my.sharepoint.com/:u:/g/personal/e16101002_stu_ahu_edu_cn/EbQz1bP2JFxHiKctQg4cwXsBYSCacwDODpoPsprYIBNm3Q?e=q5La3H)] and **Dynamic-TANet-Model** [[Onedrive](https://stuahueducn-my.sharepoint.com/:u:/g/personal/e16101002_stu_ahu_edu_cn/EaI55Lmgex5Npv0LeKsj-v0BjddW1GlnXEb0B-Ljke6Kbg?e=0N4AD4)]


get the dataset OTB2015, GOT-10k, LaSOT, UAV123, UAV20L, OxUvA from [[List](https://github.com/wangxiao5791509/DeepMTA_PyTorch/blob/master/download_links_for_tracking_datasets.txt)]. 

Download TNL2K dataset (published on CVPR 2021, 1300/700 for train and test subset) from: https://sites.google.com/view/langtrackbenchmark/


## Train: 
1. you can directly use the pre-trained tracking model of THOR [[github](https://github.com/xl-sr/THOR)]; 

2. train Dynamic Target-aware Attention: 
~~~
cd ~/DeepMTA_TCSVT_project/trackers/dcynet_modules_adaptis/ 
python train.py
~~~

3. train Trajectory Evaluation Network: 
~~~
python train_traj_measure_net.py
~~~




## Tracking:

take got-10k and LaSOT dataset as the examples: 
~~~
python testing.py -d GOT10k -t SiamRPN --lb_type ensemble

python testing.py -d LaSOT -t SiamRPN --lb_type ensemble
~~~






### Benchmark Results: 
Experimental results on the compared tracking benchmarks 

[[OTB2015]()]
[[LaSOT](https://stuahueducn-my.sharepoint.com/:u:/g/personal/e16101002_stu_ahu_edu_cn/Ec99MGQJXlJEjJFtpn7tJzoBTl77yVKt4wBOd9amXWR5lQ?e=u0eShJ)]
[[OxUvA](https://stuahueducn-my.sharepoint.com/:u:/g/personal/e16101002_stu_ahu_edu_cn/Efqz3Y2KSVdCnEl0ephudGQBNELXW7dgESWfvGmmdVVFyQ?e=D049Wf)]
[[GOT-10k](https://stuahueducn-my.sharepoint.com/:u:/g/personal/e16101002_stu_ahu_edu_cn/EbUB51geqFJEupM70SY6lfYBRkMAgKjfpH9MB6dlPKWzMg?e=kkuB6f)]
[[UAV123](https://stuahueducn-my.sharepoint.com/:u:/g/personal/e16101002_stu_ahu_edu_cn/EbhtNj6ZHRpJp34c07Qk9a4Bd522CYx4zcjOFKB6AWTUpA?e=4qEBdP)]
[[TNL2K](https://stuahueducn-my.sharepoint.com/:u:/g/personal/e16101002_stu_ahu_edu_cn/EaiGld9vweVNv6HiR3gfnlQBLlFiC29Se-MOFLJV_ooJIA?e=cXliLz)]





### Tracking Results: 

#### Tracking results on LaSOT dataset. 
![fig-1](https://github.com/wangxiao5791509/DeepMTA_PyTorch/blob/master/figures/lasot_result.png)

#### Tracking results on TNL2K dataset. 
![fig-1](https://github.com/wangxiao5791509/DeepMTA_PyTorch/blob/master/figures/benchmarkresults.png)

#### Attention prediciton and Tracking Results. 
![fig-1](https://github.com/wangxiao5791509/DeepMTA_PyTorch/blob/master/figures/attention_supplement.jpg)
![fig-1](https://github.com/wangxiao5791509/DeepMTA_PyTorch/blob/master/figures/trackingresults_vis.jpg)






### Acknowledgement:
Our tracker is developed based on **THOR** which is published on BMVC-2019 [[Paper](https://arxiv.org/pdf/1907.12920.pdf)] [[Code](https://github.com/xl-sr/THOR)]





### Citation: 
If you find this paper useful for your research, please consider to cite our paper:
~~~
@inproceedings{wang2021deepmta,
 title={Dynamic Attention guided Multi-Trajectory Analysis for Single Object Tracking},
 author={Xiao, Wang and Zhe, Chen and Jin, Tang and Bin, Luo and Yaowei, Wang and Yonghong, Tian and Feng, Wu},
 booktitle={IEEE Transactions on Circuits and Systems for Video Technology},
 doi={10.1109/TCSVT.2021.3056684}, 
 year={2021}
}
~~~

If you have any questions about this work, please contact with me via: wangx03@pcl.ac.cn or wangxiaocvpr@foxmail.com 


