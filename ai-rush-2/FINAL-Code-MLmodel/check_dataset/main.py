#from data_local_loader import get_data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import argparse
import numpy as np
import time
import datetime
import pandas as pd
import pickle

# xDeepFM model
import tensorflow as tf
#from deepctr.xdeepfm import xDeepFM
#from deepctr.inputs import SparseFeat,DenseFeat,get_fixlen_feature_names
from sklearn.metrics import mean_squared_error,accuracy_score, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_loader import feed_infer
from evaluation import evaluation_metrics
import nsml
import keras
import math

from multiprocessing import Pool
import time
from tqdm import tqdm

from Resnet_feature_extractor import resnet_feature_extractor

# new things for xgboost
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
import xgboost as xgb


if not nsml.IS_ON_NSML:
    DATASET_PATH = os.path.join('/airush2_temp')
    DATASET_NAME = 'airush2_temp'
    print('use local gpu...!')
    use_nsml = False
else:
    DATASET_PATH = os.path.join(nsml.DATASET_PATH)
    print('start using nsml...!')
    print('DATASET_PATH: ', DATASET_PATH)
    use_nsml = True

def main(args):
    
    if args.arch == 'xgboost' and args.mode == 'train':
        s = time.time()
        csv_file = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data')
        item = pd.read_csv(csv_file,
                    dtype={
                        'article_id': str,
                        'hh': int, 'gender': str,
                        'age_range': str,
                        'read_article_ids': str
                    }, sep='\t')
        label_data_path = os.path.join(DATASET_PATH, 'train',
                                os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')
        label = pd.read_csv(label_data_path,
                    dtype={'label': int},
                    sep='\t')
        item['label']  = label
        print(f'item.columns : {item.columns}')

        # 필요 데이터에 따라 article_id 를 unknown 처리 해주자.
        s = time.time()
        print(f'before test article preprocess : {len(item)}')

        ############################ make more feature !!!!!!! #################################
        ############## 1. read_article_ids len cnt -- user feature #################################################
        len_lis = []

        read_article_ids_all = item['read_article_ids'].tolist()
        for i in range(len(item)):
            li = read_article_ids_all[i]
            if type(li) == float:
                len_lis.append(0)
                continue
            len_li = len(li.split(','))
            len_lis.append(len_li)
        
        
        item['len']  = len_lis
        #############################################################################################################
        
        ################ 2. read_cnt, total_cnt, prob_read_cnt --- article feature ####################################
        read_cnt = item[item['label'] == 1].groupby('article_id').agg({'hh' : 'count'})
        read_cnt = read_cnt.reset_index()
        read_cnt = read_cnt.rename(columns = {'hh':'read_cnt'})

        read_cnt_list = read_cnt['read_cnt'].tolist()
        read_cnt_artic_list = read_cnt['article_id'].tolist()
        print(read_cnt.head(5))
        print(f'len read_cnt_list : {len(read_cnt)}')
        print(f'len read_cnt_artic_list : {len(read_cnt_artic_list)}')
        
        total_cnt = item.groupby('article_id').agg({'hh' : 'count'})
        total_cnt = total_cnt.reset_index()
        total_cnt = total_cnt.rename(columns = {'hh':'read_cnt'})
        total_cnt_list = total_cnt['read_cnt'].tolist()
        total_cnt_artic_list = total_cnt['article_id'].tolist()
        print(total_cnt.head(5))
        print(f'len read_cnt : {len(total_cnt_list)}')
        print(f'len read_cnt_artic_list : {len(total_cnt_artic_list)}')
        
        item_article_list = item['article_id'].tolist()
        item_unique_article_list = list(set(item_article_list))

        print(f'len read_cnt_list list : {len(read_cnt_list)}')
        for i in range(len(read_cnt_list)):
            print(read_cnt_list[i], end=' ')

        print()
        print(f'len read_cnt_artic_list list : {len(read_cnt_artic_list)}')
        for i in range(len(read_cnt_artic_list)):
            print(read_cnt_artic_list[i], end=' ')
        print()
        print(f'len total_cnt_list list : {len(total_cnt_list)}')
        for i in range(len(total_cnt_list)):
            print(total_cnt_list[i], end=' ')

        print()
        print(f'len total_cnt_artic_list list : {len(total_cnt_artic_list)}')
        for i in range(len(total_cnt_artic_list)):
            print(total_cnt_artic_list[i], end=' ')            

        print()
        print(f'len item unique article list : {len(item_unique_article_list)}')
        for i in range(len(item_unique_article_list)):
            print(item_unique_article_list[i], end=' ')
        
        print()

        # append rcl, rcp, tcl
        rcl = []
        rcp = []
        tcl = []

        for i in range(len(item)):
            cur_article_id = item_article_list[i]
            # read_cnt & prob
            if cur_article_id in read_cnt_artic_list:
                for j in range(len(read_cnt_artic_list)):
                    if(cur_article_id == read_cnt_artic_list[j]):
                        rcl.append(read_cnt_list[j])
                        rcp.append(read_cnt_list[j]/total_cnt_list[j])
                        break
            else :
                rcl.append(0)
                rcp.append(0.0)
                
            for j in range(len(total_cnt_artic_list)):
                if cur_article_id == total_cnt_artic_list[j]:
                    tcl.append(total_cnt_list[j])
                    break
           
                
        print(f'rcl len : {len(rcl)}')
        print(f'rcp len : {len(rcp)}')
        print(f'tcl len : {len(tcl)}')
        item['rcl'] = rcl
        item['rcp'] = rcp
        item['tcl'] = tcl


        #################################### new cols append, mean, std
        means = []
        stds = []
        for i in range(len(rcl)):
            m = float(len_lis[i]) + float(rcl[i]) + float(rcp[i]) + float(tcl[i])
            m /= 4
            s = (float(len_lis[i]) - m)**2 + (float(rcl[i]) - m)**2 + (float(rcp[i]) - m)**2 + (float(tcl[i]) - m)**2
            s /= 3

            means.append(m)
            stds.append(s)
        print(f'means len{len(means)}')
        print(f'stds len{len(stds)}')

        item['means'] = means
        item['stds'] = stds

        print('current item columns : ', item.columns.tolist())


        
        item.hh = item.hh.astype('object')

        # select feature to train
        #cols = ['article_id', 'hh', 'gender', 'age_range', 'label','len' ,'rcl', 'rcp', 'tcl']
        cols = ['hh', 'gender', 'age_range', 'label','len' ,'rcl', 'rcp', 'tcl','means','stds'] # for train
       
        item = item[cols]
        item.info()

        print('end for make feature')
        print('time : ',time.time()  -s)
        print('--------------')

    
    
    if args.pause:
        nsml.paused(scope=locals())

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)  # not work. check built_in_args in data_local_loader.py

    parser.add_argument('--train_path', type=str, default='train/train_data/train_data')
    parser.add_argument('--test_path', type=str, default='test/test_data/test_data')
    parser.add_argument('--test_tf', type=str, default='[transforms.Resize((456, 232))]')
    parser.add_argument('--train_tf', type=str, default='[transforms.Resize((456, 232))]')

    parser.add_argument('--use_sex', type=bool, default=True)
    parser.add_argument('--use_age', type=bool, default=True)
    parser.add_argument('--use_exposed_time', type=bool, default=True)
    parser.add_argument('--use_read_history', type=bool, default=False)

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--task', type=str, default='ctrpred')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_epoch_every', type=int, default=2)
    parser.add_argument('--save_step_every', type=int, default=1000)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument("--arch", type=str, default="xgboost")

    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)

    parser.add_argument('--dry_run', type=bool, default=False)

    config = parser.parse_args()
    main(config)
