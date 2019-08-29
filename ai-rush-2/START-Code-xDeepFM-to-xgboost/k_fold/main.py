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
from deepctr.xdeepfm import xDeepFM
from deepctr.inputs import SparseFeat,DenseFeat,get_fixlen_feature_names
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


fixlen_feature_names_global=[]

# bind
def bind_nsml(model, optimizer, task):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')
 
    def load(dir_name, *args, **kwargs):
        model.load_weights(os.path.join(dir_name,'model'))
        print('model loaded')

    def infer(root, phase):
        return _infer(root, phase, model=model, task=task)

    nsml.bind(save=save, load=load, infer=infer)

def _infer(root, phase, model, task):

    print('_infer root - : ', root) 
    y_pred = []
        
    return y_pred

def data_generator(df, batch_size = 2048):
    i = 0
    Y = df['label']
    X = df.drop(['label'],axis = 1)
    while True:
        for i in range(int(np.ceil(len(df)/batch_size))):
            #s = time.time()
            x_batch = X[i*batch_size:(i+1)*batch_size]
            y_batch = np.array(Y[i*batch_size:(i+1)*batch_size]).astype('float32')

            #train, test = train_test_split(x_batch, test_size=0.2,shuffle=False)
            #train_label, test_label = train_test_split(y_batch, test_size=0.2,shuffle=False)
            train_model_input = []
            for name in fixlen_feature_names_global:
                #print(name)
                # dense feature 처리도 다 해주자
                # if dense feature 면 ~  다 이렇게
                if name == 'image_feature':
                    train_model_input.append(np.array( x_batch['image_feature'].values.tolist()))
                    #print(np.array(x_batch['image_feature'].values.tolist()).shape)
                elif name == 'read_cnt_prob':
                    train_model_input.append(np.array(x_batch['read_cnt_prob'].values.tolist()))
                else:
                    train_model_input.append(x_batch[name +'_onehot'].values)

            
            #print(f'len train_model_input {len(train_model_input)}')
            #print(f'len train_label {len(y_batch)}')
            #print(f'generated batch {time.time() - s} sec')

            yield train_model_input, y_batch

def scheduler(epoch):
    if epoch < 3:
        return 0.001
    elif epoch < 30 :
        return 0.0005
    else:
        return 0.0001
    
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        print(f"epoch: {epoch}, train_acc: {logs['acc']}")
        nsml.save(str(epoch))


def main(args):
    
    if args.arch == 'xDeepFM':
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
        s = time.time()
        print(f'before test article preprocess : {len(item)}')

        sparse_features = ['article_id', 'hh','gender','age_range','len_bin']
        dense_features = ['image_feature', 'read_cnt_prob']
        target = ['label']
        
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
    
        id_to_artic = dict()
        artics = item['article_id'].tolist()
        

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

        # lit # test_article_ids list
        lit_cnt = []
        lit_total_cnt = []
        lit_cnt_prob = []
        lit = list(set(artics))
        lit.sort()
        print(lit[:10])
        print(f'len(lit):{len(lit)}')
        for i in range(len(lit)):
            # lit_cnt
            cur_artic = lit[i]
            if cur_artic not in read_cnt_artic_list :
                lit_cnt.append(0)
            else :
                for j in range(len(read_cnt_artic_list)) :
                    if cur_artic == read_cnt_artic_list[j]:
                        lit_cnt.append(read_cnt_list[j])
                        break
            # lit_total_cnt 
            if cur_artic not in total_cnt_artic_list :
                lit_total_cnt.append(0)
            else :
                for j in range(len(total_cnt_artic_list)):
                    if cur_artic == total_cnt_artic_list[j]:
                        lit_total_cnt.append(total_cnt_list[j])
                        break
            # lit_cnt_prob
            if lit_total_cnt[i] == 0:
                lit_cnt_prob.append(0)
            else :
                lit_cnt_prob.append(lit_cnt[i] / lit_total_cnt[i])
        print('--- read_cnt article feature completed ---')
        print(f'lit_cnt {len(lit_cnt)}')
        print(f'lit_total_cnt {len(lit_total_cnt)}')
        print(f'lit_cnt_prob {len(lit_cnt_prob)}')

    
        #### fea
        print('feature dict generate')
        file_list1 = os.listdir(DATASET_PATH)
        file_list2 = os.listdir(DATASET_PATH+'/train')
        file_list3 = os.listdir(DATASET_PATH+'/train/train_data')

        print(file_list1)
        print(file_list2)
        print(file_list3)
        resnet_feature_extractor(args.mode)
        
        print(file_list1)
        print(file_list2)
        print(file_list3)

        # One hot Encoding
        with open(os.path.join('train_image_features_50.pkl'), 'rb') as handle:
            image_feature_dict = pickle.load(handle)


        print('check artic feature')
        print(f"757518f4a3da : {image_feature_dict['757518f4a3da']}")
    
        lbe = LabelEncoder()
        lbe.fit(lit)
        item['article_id' + '_onehot'] = lbe.transform(item['article_id'])
        print(lbe.classes_)

        for feat in sparse_features[1:]:
            lbe = LabelEncoder()
            item[feat +'_onehot'] = lbe.fit_transform(item[feat]) # 이때 고친 라벨이 같은 라벨인지도 필수로 확인해야함

        print(item.head(10))
        print('columns name : ',item.columns)
        fixlen_feature_columns = [SparseFeat('article_id', len(lit))]
        fixlen_feature_columns += [SparseFeat(feat, item[feat +'_onehot'].nunique()) for feat in sparse_features[1:]]
        fixlen_feature_columns += [DenseFeat('image_feature',len(image_feature_dict[artics[0]]))]
        fixlen_feature_columns += [DenseFeat('read_cnt_prob',1)]
        
        print(f'fixlen_feature_columns : {fixlen_feature_columns}')
        idx_artics_all = item['article_id'+'_onehot'].tolist()
        
        for i in range(len(artics)):
            idx_artic = idx_artics_all[i]
            if idx_artic not in id_to_artic.keys():
                id_to_artic[idx_artic] = artics[i]
        
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns  
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        print(fixlen_feature_names)
        global fixlen_feature_names_global
        fixlen_feature_names_global = fixlen_feature_names
        model = xDeepFM(linear_feature_columns, dnn_feature_columns, task= 'binary')
        print('---model defined---')
        print(time.time() - s ,'seconds')

        ##### print need 

        for artic in lit:
            print(artic, end =',')
        print()
        print('new')
        print()

        print(len(lit_cnt_prob))
        for prob in lit_cnt_prob:
            prob = round(prob,4)
            print(prob, end=',')
        print()
        print('end')
        print('--------------')


    optimizer = tf.keras.optimizers.Adam(args.lr)
    s = time.time()

    # negative sampling
    item_pos = item[item['label'] == 1] 
    item_neg = item[item['label'] == 0]
    

    dn_1 = item_neg.sample(n=3*len(item_pos), random_state=42)
    dn_2 = item_neg.sample(n=3*len(item_pos), random_state=20)
    dn_3 = item_neg.sample(n=3*len(item_pos), random_state=7)
    dn_4 = item_neg.sample(n=3*len(item_pos), random_state=33)
    dn_5 = item_neg.sample(n=3*len(item_pos), random_state=41)

    dn_1.reset_index()
    
    data_1 = pd.concat([dn_1,item_pos]).sample(frac=1, random_state=42).reset_index()
    data_1_article_idxs = data_1['article_id_onehot'].tolist()
    data_1_article = data_1['article_id'].tolist()
    print(f'len data_1 : {len(data_1)}')
    print(data_1.head(5))
    li1 = []
    li2 = []
    li3 = []
    for i in range(len(data_1_article)):
        for j in range(len(lit_cnt_prob)):
            if data_1_article[i] == lit[j] :
                li3.append(lit_cnt_prob[j])
                break
    data_1['read_cnt_prob'] = li3
    print('---read_cnt_prob end---')
    ## preprocess append


    data_2 = pd.concat([dn_2,item_pos]).sample(frac=1, random_state=42).reset_index()
    data_3 = pd.concat([dn_3,item_pos]).sample(frac=1, random_state=42).reset_index()
    data_4 = pd.concat([dn_4,item_pos]).sample(frac=1, random_state=42).reset_index()
    data_5 = pd.concat([dn_5,item_pos]).sample(frac=1, random_state=42).reset_index()


    
    

    
    li = []
    for i in range(len(data_1_article_idxs)):
        image_feature = image_feature_dict[id_to_artic[data_1_article_idxs[i]]]
        li.append(image_feature)
    print(f'article_id : {data_1_article[0]}')
    print(f'article_image_feature : {image_feature_dict[data_1_article[0]]}')
    
    data_1['image_feature'] = li
    li = []
    print(f'finished data_1_image_feature : {time.time() - s} sec')



    if use_nsml:
        bind_nsml(model, optimizer, args.task)
    if args.pause:
        nsml.paused(scope=locals())

    if (args.mode == 'train') or args.dry_run:
        best_loss = 1000
        if args.dry_run:
            print('start dry-running...!')
            args.num_epochs = 1
        else:
            print('start training...!')
        
        model.compile(tf.keras.optimizers.Adam(args.lr),'mse',metrics=['accuracy'],)
        train_generator = data_generator(data_1)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        #k_fold 할때는 check point 빼자
        save_cbk = CustomModelCheckpoint()
        
        history = model.fit_generator(train_generator,
                     epochs=100, verbose=2, workers = 8, steps_per_epoch=np.ceil(len(data_1)/2048), callbacks = [lr_scheduler, save_cbk])
        print('again')
        


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
    parser.add_argument("--arch", type=str, default="xDeepFM")

    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)

    parser.add_argument('--dry_run', type=bool, default=False)

    config = parser.parse_args()
    main(config)
