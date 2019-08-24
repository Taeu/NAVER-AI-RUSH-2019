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

# expected to be a difficult problem
# Gives other meta data (gender age, etc.) but it's hard to predict click through rate
# How to use image and search history seems to be the key to problem solving. Very important data
# Image processing is key. hint: A unique image can be much smaller than the number of data.
# For example, storing image features separately and stacking them first,
# then reading them and learning artificial neural networks is good in terms of GPU efficiency.
# -> image feature has been extracted and loaded separately.
# The retrieval history is how to preprocess the sequential data and train it on which model.
# Greatly needed efficient coding of CNN RNNs.
# You can also try to change the training data set itself. Because it deals with very imbalanced problems.
# Refactor to summarize from existing experiment code.

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
        """
        state = {
            'model': model.state_dict(), # 이부분 수정
            'optimizer': optimizer.state_dict() # 이부분도 수정
        }
        torch.save(state, os.path.join(dir_name, 'model.ckpt'))
        # tf.keras.models.save_model( model, os.path.join(dir_name,'model.ckpt'))
        # tf.keras.models.save_model(    model,    filepath,    overwrite=True,    include_optimizer=True,    save_format=None)
        print('saved model checkpoints...!')
        """
    def load(dir_name, *args, **kwargs):
        model.load_weights(os.path.join(dir_name,'model'))
        print('model loaded')
        """
        state = torch.load(os.path.join(dir_name, 'model.ckpt'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        # tf.keras.models.load_model(os.path.join(dir_name, 'model.ckpt'))
        # tf.keras.models.load_model(    filepath,    custom_objects=None,    compile=True)

        print('loaded model checkpoints...!')
        """
    def infer(root, phase):
        return _infer(root, phase, model=model, task=task)

    nsml.bind(save=save, load=load, infer=infer)

def _infer(root, phase, model, task):
    # root : csv file path
    # change soon
    print('_infer root - : ', root)

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

import keras

class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, list_IDs, labels, df, batch_size=2048, dim=(32,32,32),
                 n_classes=1, ):
        'Initialization'
      
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

import math
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
                if name == 'image_feature':
                    train_model_input.append(np.array( x_batch['image_feature'].values.tolist()))
                    #print(np.array(x_batch['image_feature'].values.tolist()).shape)
                else:
                    train_model_input.append(x_batch[name].values)

            
            #print(f'len train_model_input {len(train_model_input)}')
            #print(f'len train_label {len(y_batch)}')
            #print(f'generated batch {time.time() - s} sec')

            yield train_model_input, y_batch


from multiprocessing import Pool
import time
from tqdm import tqdm


def scheduler(epoch):
  if epoch < 20:
    return 0.001
  else:
    mini_lr = 0.00001
    lr = 0.001 * math.exp(0.1 * (20 - epoch))
    
    if ( lr < mini_lr):
        return mini_lr
    else :
        print(f'current learning rate : {lr}')
        return lr

#reduce_lr = ReduceLROnPlateau(monitor='mse', factor=0.2,patience=5, min_lr=0.001)
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        print(f"epoch: {epoch}, train_acc: {logs['acc']}")
        nsml.save(str(epoch))
        #self.model.save('model.h5', overwrite=True)

def main(args):
    if args.arch == 'MLP':
        model = get_mlp(num_classes=args.num_classes)
    elif args.arch == 'Resnet':
        model = get_resnet18(num_classes=args.num_classes)
    elif args.arch == 'xDeepFM':
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
        print(len(item))
        sparse_features = ['article_id', 'hh','gender','age_range','len_bin']
        dense_features = ['image_feature']
        target = ['label']
        print(time.time() - s ,'seconds')
        s = time.time()
        len_lis = []

        read_article_ids_all = item['read_article_ids'].tolist()
        for i in range(len(item)):
            li = read_article_ids_all[i]
            if type(li) == float:
                len_lis.append(0)
                continue
            len_li = len(li.split(','))
            len_lis.append(len_li)
        print(f'read_article_ids_all len : {len(read_article_ids_all)}')
        
        """
        def extract_len_read_article(read_article_ids):
            if type(read_article_ids) == float:
                return 0
            else :
                return len(read_article_ids.split(','))
        read_article_ids_all = item['read_article_ids'].tolist()
        with Pool(processes=6) as p:
            len_lis = list(tqdm(p.imap(extract_len_read_article, read_article_ids_all), total=len(read_article_ids_all)))
        """
        item['len']  = len_lis
        item['len_bin']  = pd.qcut(item['len'],6,duplicates='drop')
        print('len_bin finished ',time.time() - s ,'seconds')
        id_to_artic = dict()
        artics = item['article_id'].tolist()
        
        with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'), 'rb') as handle:
            image_feature_dict = pickle.load(handle)
        for feat in sparse_features:
            lbe = LabelEncoder()
            item[feat] = lbe.fit_transform(item[feat])
        fixlen_feature_columns = [SparseFeat(feat, item[feat].nunique()) for feat in sparse_features]
        fixlen_feature_columns += [DenseFeat(feat,len(image_feature_dict[artics[0]])) for feat in dense_features]
        print(artics[0])
        print(fixlen_feature_columns)
        
        """
        [SparseFeat(name='article_id', dimension=1896, use_hash=False, dtype='int32', embedding_name='article_id', embedding=True), SparseFeat(name='hh', dimension=24, use_hash=False, dtype='int32', embedding_name='hh', embedding=True), SparseFeat(name='gender', dimension=2, use_hash=False, dtype='int32', embedding_name='gender', embedding=True), SparseFeat(name='age_range', dimension=9, use_hash=False, dtype='int32', embedding_name='age_range', embedding=True), SparseFeat(name='len_bin', dimension=5, use_hash=False, dtype='int32', embedding_name='len_bin', embedding=True), DenseFeat(name='image_feature', dimension=2048, dtype='float32')]
        
        """
        print('---fixlen_feature_columns finished---')
        s = time.time()
        idx_artics_all = item['article_id'].tolist()
        print(f'idx_artics_all len : {len(idx_artics_all)}')
        print(f'artics len : {len(artics)}')
        for i in range(len(artics)):
            idx_artic = idx_artics_all[i]
            if idx_artic not in id_to_artic.keys():
                id_to_artic[idx_artic] = artics[i]
        print(f'id_to_artic len : {len(id_to_artic)}')
        print(time.time() - s ,'seconds')
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

    """
    if args.use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()

    """
    optimizer = tf.keras.optimizers.Adam(args.lr)
    
    # negative sampling
    item_pos = item[item['label'] == 1] 
    item_neg = item[item['label'] == 0]
    print(f'len item_pos : {len(item_pos)}')
    print(f'len item_neg : {len(item_neg)}')

    dn_1 = item_neg.sample(n=2*len(item_pos), random_state=42)
    dn_1.reset_index()
    print(f'len dn_1 : {len(dn_1)}')

    data_1 = pd.concat([dn_1,item_pos]).sample(frac=1, random_state=42).reset_index()
    print(f'len data_1 : {len(data_1)}')
    print('--- negative sampling completed ---')

    s = time.time()
    data_1_article_idxs = data_1['article_id'].tolist()
    li = []
    for i in range(len(data_1_article_idxs)):
        image_feature = image_feature_dict[id_to_artic[data_1_article_idxs[i]]]
        li.append(image_feature)

    print(f'len image_feature : {len(li)}')
    data_1['image_feature'] = li
    li = []
    print(f'finished data_1_image_feature : {time.time() - s} sec')

    print(f'generate all x_train')


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
        # 미리 전체를 다 만들어놓자 굳이 generator 안써도 되겠네
        model.compile(tf.keras.optimizers.Adam(args.lr),'mse',metrics=['accuracy'],)
        train_generator = data_generator(data_1)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        save_cbk = CustomModelCheckpoint()

        history = model.fit_generator(train_generator,
                     epochs=200, verbose=2, workers = 8, steps_per_epoch=np.ceil(len(data_1)/2048), callbacks = [lr_scheduler,save_cbk])
        print('again')
        
        """
        for epoch in range(args.num_epochs):
            for i, data in enumerate(train_loader):
                
                print(type(pd1))
                print(pd1)
                
                images = images.cuda()
                extracted_image_features = extracted_image_features.cuda()
                flat_features = flat_features.cuda()
                labels = labels.cuda()

                # forward
                if args.arch == 'MLP':
                    logits = model(extracted_image_features, flat_features)
                elif args.arch == 'Resnet':
                    logits = model(images, flat_features)
                criterion = nn.MSELoss()
                loss = torch.sqrt(criterion(logits.squeeze(), labels.float()))

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss < best_loss:
                    nsml.save('best_loss')  # this will save your best model on nsml.

                if i % args.print_every == 0:
                    elapsed = datetime.datetime.now() - start_time
                    print('Elapsed [%s], Epoch [%i/%i], Step [%i/%i], Loss: %.4f'
                          % (elapsed, epoch + 1, args.num_epochs, i + 1, iter_per_epoch, loss.item()))
                if i % args.save_step_every == 0:
                    # print('debug ] save testing purpose')
                    nsml.save('step_' + str(i))  # this will save your current model on nsml.
                
            if epoch % args.save_epoch_every == 0:
                nsml.save('epoch_' + str(epoch))  # this will save your current model on nsml.


    nsml.save('final')
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8)  # not work. check built_in_args in data_local_loader.py

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
