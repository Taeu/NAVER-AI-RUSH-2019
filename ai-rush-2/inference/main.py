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
def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')
 
    def load(dir_name, *args, **kwargs):
        model.load_weights(os.path.join(dir_name,'model'))
        print('model loaded')

    def infer(root, phase):
        return _infer(root, phase, model=model )

    nsml.bind(save=save, load=load, infer=infer)

def _infer(root, phase, model,):
    # root : csv file path
    # change soon
    print('_infer root - : ', root)
    checkpoint_session = ['401','team_62/airush2/176']
    #model, fixlen_feature_names_global, item = get_xDeepFM()
    global fixlen_feature_names_global
    model, fixlen_feature_names_global, item, image_feature_dict, id_to_artic = get_item()
    bind_nsml(model)
    #bind_nsml(model, [], args.task)
    print('--get item finished---')
    nsml.load(checkpoint = str(checkpoint_session[0]), session = str(checkpoint_session[1]))
    print('-- model_load completed --')

    s = time.time()
    data_1_article_idxs = item['article_id'].tolist()
    li = []

    
    for i in range(len(data_1_article_idxs)):
        image_feature = image_feature_dict[id_to_artic[data_1_article_idxs[i]]]
        li.append(image_feature)

    
    item['image_feature'] = li
    li = []
    print(f'finished data_1_image_feature : {time.time() - s} sec')
    test_generator = data_generator_test(item)

    # 맞확
    predicts = model.predict_generator(test_generator, workers = 8, use_multiprocessing = True)
    print(f'y_pred shape : {predicts.shape}')
    return predicts
    """
    with torch.no_grad():
        model.eval()
        test_loader, dataset_sizes = get_data_loader(root, phase)
        y_pred = []
        print('start infer')
        for i, data in enumerate(test_loader):
            images, extracted_image_features, labels, flat_features = data

            # images = images.cuda()
            extracted_image_features = extracted_image_features.cuda()
            flat_features = flat_features.cuda()
            # labels = labels.cuda()

            logits = model(extracted_image_features, flat_features)
            y_pred += logits.cpu().squeeze().numpy().tolist()

        print('end infer')
    return y_pred
    """

def data_generator_test(df, batch_size = 1):
    i = 0
    X = df
    while True:
        for i in range(int(np.ceil(len(df)/batch_size))):
            #s = time.time()
            x_batch = X[i*batch_size:(i+1)*batch_size]
            

            test_model_input = []
            for name in fixlen_feature_names_global:
                #print(name)
                if name == 'image_feature':
                    test_model_input.append(np.array( x_batch['image_feature'].values.tolist()))
                    #print(np.array(x_batch['image_feature'].values.tolist()).shape)
                else:
                    test_model_input.append(x_batch[name].values)

            yield test_model_input

def scheduler(epoch):
    if epoch < 200:
        return 0.001
    elif epoch < 400 :
        return 0.0005
    elif epoch < 700 :
        return 0.0001
    

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        print(f"epoch: {epoch}, train_acc: {logs['acc']}")
        nsml.save(str(epoch))

def get_xDeepFM():

    csv_file = os.path.join(DATASET_PATH, 'test', 'test_data', 'test_data')
    item = pd.read_csv(csv_file,
                dtype={
                    'article_id': str,
                    'hh': int, 'gender': str,
                    'age_range': str,
                    'read_article_ids': str
                }, sep='\t')
      
    sparse_features = ['article_id', 'hh','gender','age_range','len_bin']
    dense_features = ['image_feature']
    target = ['label']
    
    
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
    
    with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'), 'rb') as handle:
        image_feature_dict = pickle.load(handle)

    print('image_feaeture_dict loaded..')
    for feat in sparse_features:
        lbe = LabelEncoder()
        item[feat] = lbe.fit_transform(item[feat])

    # test set으로 구성해도 되고 item 을..

    fixlen_feature_columns = [SparseFeat(feat, item[feat].nunique()) for feat in sparse_features]
    fixlen_feature_columns += [DenseFeat(feat,len(image_feature_dict[artics[0]])) for feat in dense_features]
    print(fixlen_feature_columns)
    
    
    idx_artics_all = item['article_id'].tolist()
    
    for i in range(len(artics)):
        idx_artic = idx_artics_all[i]
        if idx_artic not in id_to_artic.keys():
            id_to_artic[idx_artic] = artics[i]
    
    
       
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns  
    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
    
    fixlen_feature_names_global = fixlen_feature_names
    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task= 'binary')
    return model, fixlen_feature_names_global, item

def get_item():
    csv_file = os.path.join(DATASET_PATH, 'test', 'test_data', 'test_data')
    item = pd.read_csv(csv_file,
                dtype={
                    'article_id': str,
                    'hh': int, 'gender': str,
                    'age_range': str,
                    'read_article_ids': str
                }, sep='\t')
      
    sparse_features = ['article_id', 'hh','gender','age_range','len_bin']
    dense_features = ['image_feature']
    target = ['label']

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
    
    with open(os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image_features.pkl'), 'rb') as handle:
        image_feature_dict = pickle.load(handle)

    print('image_feaeture_dict loaded..')
    for feat in sparse_features:
        lbe = LabelEncoder()
        item[feat] = lbe.fit_transform(item[feat])

    # test set으로 구성해도 되고 item 을..

    fixlen_feature_columns = [SparseFeat(feat, item[feat].nunique()) for feat in sparse_features]
    fixlen_feature_columns += [DenseFeat(feat,len(image_feature_dict[artics[0]])) for feat in dense_features]
    
    print(fixlen_feature_columns)
    
    
    idx_artics_all = item['article_id'].tolist()
    
    for i in range(len(artics)):
        idx_artic = idx_artics_all[i]
        if idx_artic not in id_to_artic.keys():
            id_to_artic[idx_artic] = artics[i]
    
    
       
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns  
    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
    
    fixlen_feature_names_global = fixlen_feature_names

    model = xDeepFM(linear_feature_columns, dnn_feature_columns, task= 'binary')
    #bind_nsml(model, list(), args.task)

    return model, fixlen_feature_names_global, item,image_feature_dict, id_to_artic


def main(args):
    
    if args.arch == 'xDeepFM' and args.mode == 'train':
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
        
        sparse_features = ['article_id', 'hh','gender','age_range','len_bin']
        dense_features = ['image_feature']
        target = ['label']
        
        
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
        
        with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'), 'rb') as handle:
            image_feature_dict = pickle.load(handle)
        for feat in sparse_features:
            lbe = LabelEncoder()
            item[feat] = lbe.fit_transform(item[feat])
        fixlen_feature_columns = [SparseFeat(feat, item[feat].nunique()) for feat in sparse_features]
        fixlen_feature_columns += [DenseFeat(feat,len(image_feature_dict[artics[0]])) for feat in dense_features]
        
        
        
        idx_artics_all = item['article_id'].tolist()
        
        for i in range(len(artics)):
            idx_artic = idx_artics_all[i]
            if idx_artic not in id_to_artic.keys():
                id_to_artic[idx_artic] = artics[i]
        
       
            #image_feature_dict[article_id] 로 가져오면 되니까 일단 패스
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns  
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        print(fixlen_feature_names)
        global fixlen_feature_names_global
        fixlen_feature_names_global = fixlen_feature_names
        model = xDeepFM(linear_feature_columns, dnn_feature_columns, task= 'binary')
        print('---model defined---')
        # 만들었던 파일들 저장하는 것도 하나 짜기, 매번 돌릴 수 없으니까
        print(time.time() - s ,'seconds')

    if use_nsml and args.mode == 'train':
        bind_nsml(model)

    if (args.mode == 'train'):
        if args.dry_run:
            print('start dry-running...!')
            args.num_epochs = 1
        else:
            print('start training...!')
        # 미리 전체를 다 만들어놓자 굳이 generator 안써도 되겠네

        nsml.save('infer')
        print('end')
    print('end_main')


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
