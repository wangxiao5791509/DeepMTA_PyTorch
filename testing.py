import argparse
import numpy as np
from types import SimpleNamespace
import json

from benchmark.vot import test_vot, eval_vot
from benchmark.otb import test_otb, eval_otb
from benchmark.got10k import test_got
# from benchmark.got10ktrainval import test_gottrainval 
from benchmark.lasot import test_lasot, eval_lasot
# from benchmark.uav20l import test_uav20l, eval_uav20l
from benchmark.uav123 import test_uav123, eval_uav123
# from benchmark.oxuva import test_oxuva 
# from benchmark.tc128 import test_tc128, eval_tc128 

from trackers.tracker import SiamFC_Tracker, SiamRPN_Tracker, SiamMask_Tracker
from benchmark.bench_utils.benchmark_helper import load_dataset
import warnings
warnings.filterwarnings("ignore")
import ast 
import pdb 

parser = argparse.ArgumentParser(description='Test Trackers on Benchmarks.')
parser.add_argument('-d', '--dataset', dest='dataset', default='OTB2015', 
                    help='Dataset on which the benchmark is run [VOT2018, OTB2015, GOT10k, LaSOT, UAV20L]')
parser.add_argument('-t', '--tracker', dest='tracker', default='SiamRPN', 
                    help='Name of the tracker [SiamFC, SiamRPN, SiamMask]')
parser.add_argument('--vanilla', action='store_true',
                    help='Run the tracker without THOR')
parser.add_argument('-v', '--viz', action='store_true', default=False,                
                    help='Show the tracked scene, the stored templated and the modulated view')
parser.add_argument('--verbose', action='store_true',
                    help='Print additional info about THOR')
parser.add_argument('--lb_type', type=str, default='dynamic',
                    help='Specify the type of lower bound [dynamic, ensemble]')
parser.add_argument('--spec_video', type=str, default='', 
                    help='Pick a specific video by name, e.g. "lemming" on OTB2015')
parser.add_argument('--save_path', dest='save_path', default='Tracker',
                    help='Name where the tracked trajectory is stored')

def load_cfg(args):
    json_path = f"configs/{args.tracker}/"
    json_path += f"{args.dataset}_"
    if args.vanilla:
        json_path += "vanilla.json"
    else:
        json_path += f"THOR_{args.lb_type}.json"

    # pdb.set_trace() 

    cfg = json.load(open(json_path))
    return cfg


def run_bench(delete_after=False):
    args = parser.parse_args()

    cfg = load_cfg(args)
    cfg['THOR']['viz'] = args.viz
    cfg['THOR']['verbose'] = args.verbose
    
    # setup tracker and dataset
    if args.tracker == 'SiamFC':
        tracker = SiamFC_Tracker(cfg)
    elif args.tracker == 'SiamRPN':
        tracker = SiamRPN_Tracker(cfg)
    elif args.tracker == 'SiamMask':
        tracker = SiamMask_Tracker(cfg)
    else:
        raise ValueError(f"Tracker {args.tracker} does not exist.")



    dataset = load_dataset(args.dataset)
    # optionally filter for a specific videos
    if args.spec_video:

        # pdb.set_trace() 
        dataset = {args.spec_video: dataset[args.spec_video]}

    if args.dataset=="VOT2018":
        test_bench, eval_bench = test_vot, eval_vot
    elif args.dataset=="OTB2015":
        test_bench, eval_bench = test_otb, eval_otb
    elif args.dataset=="GOT10k":
        test_bench = test_got 
    elif args.dataset=="GOT10k_train_val":
        test_bench = test_gottrainval 
    elif args.dataset=="LaSOT":
        test_bench, eval_bench = test_lasot, eval_lasot
    elif args.dataset=="UAV20L":  
        test_bench, eval_bench = test_uav20l, eval_uav20l
    elif args.dataset=="UAV123":  
        test_bench, eval_bench = test_uav123, eval_uav123        
    elif args.dataset=="OXUVA": 
        test_bench = test_oxuva 
    elif args.dataset=="TC128":
        test_bench, eval_bench = test_tc128, eval_tc128 
    else:
        raise NotImplementedError(f"Procedure for {args.dataset} does not exist.")

    # testing
    total_lost = 0
    speed_list = []

    if args.dataset=="OTB2015":
        print("==>> No processing for the json file ... ")
    else: 
        dataset = ast.literal_eval(dataset) 
    # pdb.set_trace()
        
    for v_id, video in enumerate(dataset.keys(), start=1):
        tracker.temp_mem.do_full_init = True
        speed = test_bench(v_id, tracker, dataset[video], args)
        speed_list.append(speed)


    if args.dataset=="GOT10k": 
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("==>> Please evaluate online for GOT10k dataset ... ")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")        
    elif args.dataset=="OxUvA":
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")        
        print("==>> Please evaluate online for OxUvA dataset ... ")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")        
    else: 
        # evaluation
        # pdb.set_trace() 
        bench_res = eval_bench(args.save_path, delete_after)
        print(bench_res)  
        mean_fps = np.mean(np.array(speed_list))
        bench_res['mean_fps'] = mean_fps              
        print(bench_res)  

        return bench_res



if __name__ == '__main__':
    run_bench()

