import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import utils
from data_local_loader import get_data_loader

import os
import nsml
import datetime

if not nsml.IS_ON_NSML:
    DATASET_PATH = os.path.join('/home/kwpark_mk2/airush2_temp')
    DATASET_NAME = 'airush2_temp'
    print('use local gpu...!')
    use_nsml = False
else:
    DATASET_PATH = os.path.join(nsml.DATASET_PATH)
    print('start using nsml...!')
    print('DATASET_PATH: ', DATASET_PATH)
    use_nsml = True

from torch.autograd import Variable

def resnet_feature_extractor(phase) :
    resnet50 = models.resnet50(pretrained=True)
    modules = list(resnet50.children())[:-1]
    resnet50 = nn.Sequential(*modules)

    model = resnet50
    model = model.cuda()
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for p in resnet50.parameters():
        p.requires_grad = False

    if phase == 'train':
        print('load train data')
        file_list = os.listdir(DATASET_PATH+'/train/train_data/train_image')
    else:
        print('load test data')
        file_list = os.listdir(DATASET_PATH+'/test/test_data/test_image')
    print('start extracting...!')

    y_pred_dict = {}
    for fname in file_list:

        img_name = os.path.join(DATASET_PATH, phase, phase + '_data', phase + '_image', fname)

        image = utils.default_loader(img_name)
        data_transforms = utils.get_transforms('[transforms.Resize((456, 232))]', verbose=False)
        image = data_transforms['train'](image)
        image = image.unsqueeze(0)
        image = image.cuda()

        # forward
        logits = model(image)
        y_pred_dict[fname[:-4]] = logits.cpu().squeeze().numpy()

        if len(y_pred_dict) % 100 == 0:
            print('current stack size :  ', len(y_pred_dict), round(len(y_pred_dict) / len(file_list), 2) * 100, '%')

    print('extraction is done')
    dict_save_name = phase + '_image_features_50.pkl'

    with open(dict_save_name, 'wb') as handle:
        pickle.dump(y_pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done')
