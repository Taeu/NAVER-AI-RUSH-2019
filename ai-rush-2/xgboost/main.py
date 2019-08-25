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
from matplotlib import pyplot as plt
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

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

# bind
def bind_nsml(model, optimizer, task):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        pickle.dump(model, open(os.path.join(dir_name,"pickle.dat"),"wb"))
        print('model saved!')
 
    def load(dir_name, *args, **kwargs):
        model = pickle.load(open(os.path.join(dir_name,"pickle.dat"), "rb"))
        print('model loaded')

    def infer(root, phase):
        return _infer(root, phase, model=model, task=task)

    nsml.bind(save=save, load=load, infer=infer)

def _infer(root, phase, model, task):

    print('_infer root - : ', root) 
    y_pred = []
        
    return y_pred

def main(args):
    
    if args.arch == 'xgboost':
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
        item['len_bin']  = pd.qcut(item['len'],6,duplicates='drop')
 

        ################ 2. read_cnt, total_cnt, prob_read_cnt --- article feature ####################################
        read_cnt = item[item['label'] == 1].groupby('article_id').agg({'hh' : 'count'})
        read_cnt = read_cnt.reset_index()
        read_cnt = read_cnt.rename(columns = {'hh':'read_cnt'})

        read_cnt_list = read_cnt['read_cnt'].tolist()
        read_cnt_artic_list = read_cnt['article_id'].tolist()
        print(f'len read_cnt : {len(read_cnt)}')
        print(read_cnt.head(3))
        
        total_cnt = item.groupby('article_id').agg({'hh' : 'count'})
        total_cnt = total_cnt.reset_index()
        total_cnt = total_cnt.rename(columns = {'hh':'read_cnt'})
        total_cnt_list = total_cnt['read_cnt'].tolist()
        total_cnt_artic_list = total_cnt['article_id'].tolist()
        print(f'len read_cnt : {len(total_cnt)}')
        print(total_cnt.head(3))

        print('add unknown data')
        a = pd.DataFrame({'article_id' : ['unknown'], 'hh' : ['unknown'], 'gender': ['unknown'],'age_range' : ['unknown'] ,'label' : [0], 'len' : [0]},index=[len(item)])
        item = item.append(a)
        print(item.tail(1))

        # inference 시에 train 정해둔 article 로 돌아야... 아니지
        # total_cnt_artic_list, read_cnt_artic_list 가 같이 train 에 저장된 inform 불러와서 update 해야겠지
        item_article_list = item['article_id'].tolist()
        
        rcl = []
        rcp = []
        tcl = []
        dict_total_cnt_artic_list = {}
        dict_read_cnt_artic_list = {}

        for i in range(len(item)):
            cur_article_id = item_article_list[i]
            # read_cnt & prob
            if cur_article_id in total_cnt_artic_list:
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
                
            else : 
                # 생전 처음 보는 아티클, 이 경우는 그냥 평균 넣어주자
                print(cur_article_id, 'is not in train dataset')
                rcl.append(4)
                rcp.append(0.06)
                tcl.append(int(np.mean(total_cnt_list)))
                
        print(f'rcl len : {len(rcl)}')
        print(f'rcp len : {len(rcp)}')
        print(f'tcl len : {len(tcl)}')
        item['rcl'] = rcl
        item['rcp'] = rcp
        item['tcl'] = tcl

        #### feature
        print('feature dict generate')
        file_list1 = os.listdir(DATASET_PATH)
        file_list2 = os.listdir(DATASET_PATH+'/train')
        file_list3 = os.listdir(DATASET_PATH+'/train/train_data')

        resnet_feature_extractor(args.mode)

        # One hot Encoding
        with open(os.path.join('train_image_features_50.pkl'), 'rb') as handle:
            image_feature_dict = pickle.load(handle)

        print('check artic feature')
        print(f"757518f4a3da : {image_feature_dict['757518f4a3da']}")
    


        # print check all
        print('after all preprocess extract feature')
        print(item.columns)
        item.hh = item.hh.astype('object')

        # select feature to train
        cols = ['article_id', 'hh', 'gender', 'age_range', 'label','len' ,'rcl', 'rcp', 'tcl']
        item = item[cols]
        item.info()

        print('end for make feature')
        print('time : ',time.time()  -s)
        print('--------------')

    
    
    if args.pause:
        nsml.paused(scope=locals())

    s = time.time()
    print('---------train start---------')
    y = item['label']
    x = item.drop(['label'],axis = 1)
    one_hot_encoded_X = pd.get_dummies(x)
    print("# of columns after one-hot encoding: {0}".format(len(one_hot_encoded_X.columns)))
   
    params = {
    'colsample_bynode': 0.8,
    'learning_rate': 1,
    'max_depth': 5,
    'num_parallel_tree': 500,
    'objective': 'binary:logistic',
    'subsample': 0.8,
    'tree_method': 'gpu_hist'
    }

    if (args.mode == 'train') or args.dry_run:
        best_loss = 1000
        if args.dry_run:
            print('start dry-running...!')
            args.num_epochs = 1
        else:
            print('start training...!')
        # https://www.kaggle.com/infinitewing/k-fold-cv-xgboost-example-0-28
        # metric : 
        kf = KFold(n_splits=5, random_state = 42, shuffle =True)
        cvi = 0
        for train_index, test_index in kf.split(one_hot_encoded_X, y):
            
            cvi += 1
            print(cvi,' cv')
            train_x = one_hot_encoded_X.iloc[train_index]
            train_y = y.iloc[train_index]
            valid_x = one_hot_encoded_X.iloc[test_index]
            valid_y = y.iloc[test_index]
            
            print(train_x.shape)
            print(valid_x.shape)


            xgb_model = None
            # good example batch training : https://github.com/dmlc/xgboost/issues/2970
            batch_size = 100000
            d_valid = xgb.DMatrix(valid_x[0:batch_size], valid_y[0:batch_size])
            for k in range(0, (train_x.shape[0] // batch_size)*batch_size, batch_size) :
                d_train = xgb.DMatrix(train_x[k:k+batch_size], train_y[k:k+batch_size])
                
                # 'gpu_id': 0 , 'tree_method': 'gpu_hist', 'predictor':'gpu_predictor', 'max_bin': 16
                xgb_params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'map', 'seed': 99, 'silent': True ,}
                # cv 마다 다른 모델로 추가하기?
                watchlist = [(d_train, 'train'), (d_valid, 'valid')]

            
                xgb_model = xgb.train(xgb_params, d_train, 100,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=10, early_stopping_rounds=100, xgb_model = xgb_model)
                if use_nsml and k == 0:
                    bind_nsml(xgb_model, [], args.task)
                    print('bind model')
                nsml.save(str(cvi)+'_'+str(k))
            
            """
            xgb_pred = xgb_model.predict(d_test)
            xgb_preds.append(list(xgb_pred))
            
            """
    

    # if you use feature, then implement more the below..
        article_id_list = item['article_id'].tolist()
        li = []
        for i in range(len(article_id_list)):
            image_feature = image_feature_dict[article_id_list[i]]
            li.append(image_feature)
        print(f'article_id : {article_id_list[0]}')
        print(f'article_image_feature : {image_feature_dict[article_id_list[0]]}')
        
        data_1['image_feature'] = li
        li = []
        print(f'finished data_1_image_feature : {time.time() - s} sec')
        

        


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
