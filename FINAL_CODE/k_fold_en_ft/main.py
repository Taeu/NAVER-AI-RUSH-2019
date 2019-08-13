import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from model import Baseline, Resnet
import nsml
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import AIRushDataset
from model_efficientnet import EfficientNet
from torch.optim.lr_scheduler import StepLR
#from torchsummary import summary
# need for k-fold validation


import pickle as pkl
from PIL import Image

from tqdm import tqdm
from nsml import DATASET_PATH
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import time

import xgboost as xgb



validation_split = 0.1
random_seed = 42

def to_np(t):
    return t.cpu().detach().numpy()

def bind_model(model_nsml):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = {
                    'model': model_nsml.state_dict(),
                }
        torch.save(state, save_state_path)

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = torch.load(save_state_path)
        model_nsml.load_state_dict(state['model'])
        
    def infer(test_image_data_path, test_meta_data_path):
        # DONOTCHANGE This Line
        test_meta_data = pd.read_csv(test_meta_data_path, delimiter=',', header=0)
        
        input_size=224 # you can change this according to your model.
        batch_size=100 # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        device = 0
        
        dataloader = DataLoader(
                        AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                                      transform=transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True)
        
        model_nsml.to(device)
        model_nsml.eval()
        predict_list = []
        for batch_idx, image in enumerate(dataloader):
            image = image.to(device)
            output = model_nsml(image).double()
            
            output_prob = F.softmax(output, dim=1)
            predict = np.argmax(to_np(output_prob), axis=1)
            predict_list.append(predict)
                
        predict_vector = np.concatenate(predict_list, axis=0)
        return predict_vector # this return type should be a numpy array which has shape of (138343)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

class AIRushDataset(Dataset):
    def __init__(self, image_data_path, meta_data, label_path=None, transform=None, train_mode=False,label_matrix =None):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_path = label_path
        self.transform = transform
        if self.label_path is not None:
            self.label_matrix = np.load(label_path)
            if label_matrix is not None:
                self.label_matrix = label_matrix
                print('load label matrix')
                print(self.label_matrix.shape)
                print('-------------')
    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir , str(self.meta_data['package_id'].iloc[idx]) , str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load() # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3]) # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)
       
        if self.label_path is not None:
            tags = torch.tensor(np.argmax(self.label_matrix[idx])) # here, we will use only one label among multiple labels.
            return new_img, tags
        else:
            return new_img


def train_dataloader(train_meta_data,
                    input_size=128,
                    batch_size=64,
                    num_workers=4,
                    label_matrix = None,
                    transforms_i = None
                    ):
    
    image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images') 
    train_label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
    train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
    
    

    if transforms_i is None:
        transforms_i = transforms.Compose([
                      transforms.Resize((input_size, input_size)), 
                      
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(20),

                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                      ]) 
    # transforms - batch normalization add       
    dataloader = DataLoader(
               AIRushDataset(image_dir, train_meta_data, label_path=train_label_path, 
                      transform=transforms_i,
                      label_matrix = label_matrix,
                      ),        
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    return dataloader


transforms_1 = transforms.Compose([
    transforms.Resize((224, 224)), 
    
    transforms.RandomRotation(20),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]) 

transforms_2 = transforms.Compose([
    transforms.Resize((224, 224)), 
    
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]) 

transforms_3 = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]) 

transforms_4 = transforms.Compose([
    transforms.CenterCrop(224), 

    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]) 
    

def valid_dataloader(train_meta_data,
                    input_size=128,
                    batch_size=64,
                    num_workers=4,
                    label_matrix =None,
                    ):
    
    image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images') 
    train_label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
    train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
    
    


    # transforms - batch normalization add       
    dataloader = DataLoader(
               AIRushDataset(image_dir, train_meta_data, label_path=train_label_path, 
                      transform=transforms.Compose([
                      transforms.Resize((input_size, input_size)), 
                      

                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                      ]),
                      label_matrix = label_matrix
                      ),        
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    return dataloader


def train_xgb_cnn(model, x_train_image, y_train, x_valid_image, y_valid, cnn=None, num_round=100):
    # model 
    # x_train_image (batch_size, (image_size))
    # y_train : tags (batch_size,) 
    param = {'eta':0.1,
             'max_depth':6,
             'objective':'multi:softmax',
             'n_estimators':500,
             'silent':1,
             'num_class':350}

    
    with torch.no_grad():
        feat_out = model.extract_features(x_train_image)
        feat_out = model.extract_features(x_valid_image)

    dtrain = xgb.DMatrix(feat_out,
                             label=y_train)
    dvalid = xgb.DMatrix(feat_out_valid, label = y_valid)
    xgb_feature_layer = xgb.train(param, dtrain, num_round, dvalid)

    # predict 은 xgb_feature_layer.predict( test_data, num_iteration = xgb_feature_layer.best_iteration)
    # print(predict)
    return xgb_feature_layer

def get_extracted_data(model, data_loader):
    for bi, (image, tags) in enumerate(data_loader):
        print(".", end="")
        img_tensor = image.to(device)
        target = tags.to(device)
        with torch.no_grad(): feature =  model.extract_features(img_tensor)
        feature = feature.cpu().detach().squeeze(0).numpy()
        target = target.cpu().detach().squeeze(0).numpy()
        if bi == 0 :
            features = feature 
        
            targets = target 
        else :
            features = np.concatenate([features, feature], axis=0)
            targets = np.concatenate([targets, target], axis=0)
    print("")
    return features, targets
    

# dropout ratio  : 0.3 ì¼ë¡ ì¡°ì 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    
    # custom args
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--resnet', default=True)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=350) # Fixed
    parser.add_argument('--epochs', type=int, default=6) # change
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=5e-4) # change
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    if args.resnet:
        assert args.input_size == 224
        #model = Resnet(args.output_size)
        print('!!!!!!!!!!!!!!!!efficientnet load!!!!!!!!!!!!!!!!')
        model_name = 'efficientnet-b0'
        print(model_name)
        
        model = EfficientNet.from_name(model_name)
        
        #model = EfficientNet.from_pretrained(model_name, num_classes=350)
        #summary(model,input_size=(3,224,224))
    else:
        model = Baseline(args.hidden_size, args.output_size)

    # DONOTCHANGE: They are reserved for nsml


    if args.pause:
        nsml.paused(scope=locals())
    
    if args.mode == "train":
        # Warning: Do not load data before this line
        
        #skf = StratifiedKFold(n_splits = 5, random_state = random_seed)
        kf = KFold(n_splits = 4,random_state = random_seed,shuffle =True)
        image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images') 
        train_label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
        train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
        train_meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)
        label_matrix = np.load(train_label_path)
        print(len(train_meta_data))
        


        ensemble = [['team_62/airush1/415', '03'],['team_62/airush1/415','13'],['team_62/airush1/415','23'],['team_62/airush1/415','33']]
            
        for cvf in range(4):
            if cvf == 0:
                transforms_i = transforms_1
            elif cvf == 1:
                transforms_i = transforms_2
            elif cvf == 2:
                transforms_i = transforms_3
            elif cvf == 3:
                transforms_i = transforms_4


            
            for i, (train_idx, valid_idx) in enumerate(kf.split(train_meta_data)):
                if i == cvf :
                    break
            print(cvf)
            bind_model(model)            
            nsml.load(checkpoint=str(ensemble[i][1]),session=str(ensemble[i][0]))
            print('model load finished')

            
            # extract feature
            optimizer = optim.Adam(model.parameters(), args.learning_rate)
            scheduler = StepLR(optimizer, step_size=2, gamma=0.4)
            criterion = nn.CrossEntropyLoss() #multi-class classification task
            model = model.to(device)
            model.eval()
            #def train_xgb_cnn(model, x_train_image, y_train, x_valid_image, y_valid, cnn=None, num_round=100):

            meta_data_train , label_train = train_meta_data.iloc[train_idx] , label_matrix[train_idx]
            meta_data_val, label_val = train_meta_data.iloc[valid_idx], label_matrix[valid_idx]
            print('data split completed')

            dataloader_train = train_dataloader(meta_data_train, args.input_size, args.batch_size, args.num_workers, label_matrix = label_train ,transforms_i = transforms_i)
            dataloader_valid = valid_dataloader(meta_data_val, args.input_size, args.batch_size, args.num_workers, label_matrix = label_val )
            print('data load completed')
            
            features_train, targets_train = get_extracted_data(model,dataloader_train)
            features_eval, targets_eval = get_extracted_data(model,dataloader_valid)
            print(features_train.shape)
            print(targets_train.shape)
            print('features data completed')

            s = time.time()
            

            param = {'eta':0.1,
                    'random_state' : 42,
                    'max_depth':6,
                    'objective':'multi:softmax',
                    'n_estimators':500,
                    'silent':1,
                    'num_class':350}

            xgb_model = xgb.XGBClassifier(**param)
            xgb_model = xgb_model.fit(features_train,targets_train.reshape(-1), eval_set = [(features_eval, targets_eval.reshape(-1))], early_stopping_rounds = 20,)

            print('xgb finished : ',time.time() - s,'s')

            

            nsml.save(str(cvf))

            print('saved')





        print('end')


