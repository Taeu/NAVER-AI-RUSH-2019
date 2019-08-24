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

resnet50 = models.resnet50(pretrained=True)
modules = list(resnet50.children())[:-1]
resnet50 = nn.Sequential(*modules)
for p in resnet50.parameters():
    p.requires_grad = False


def main(args):
    model = resnet50

    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        print('load train data')

        file_list = os.listdir('./train/train_data/train_image')

    else:
        print('load test data')
        file_list = os.listdir('./test/test_data/test_image')

    start_time = datetime.datetime.now()
    print('start extracting...!')

    y_pred_dict = {}
    for fname in file_list:

        img_name = os.path.join(DATASET_PATH, args.mode, args.mode + '_data', args.mode + '_image', fname)

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
    dict_save_name = args.mode + '_image_features.pkl'

    with open(dict_save_name, 'wb') as handle:
        pickle.dump(y_pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--train_path', type=str, default='train/train_data/train_data')
    parser.add_argument('--test_path', type=str, default='test/test_data/test_data')
    parser.add_argument('--test_tf', type=str, default='[transforms.Resize((456, 232))]')
    parser.add_argument('--train_tf', type=str, default='[transforms.Resize((456, 232))]')

    parser.add_argument('--use_sex', type=bool, default=True)
    parser.add_argument('--use_age', type=bool, default=True)
    parser.add_argument('--use_exposed_time', type=bool, default=True)
    parser.add_argument('--use_read_history', type=bool, default=False)

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--task', type=str, default='ctrpred')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=100)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument("--arch", type=str, default="MLP")

    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)

    config = parser.parse_args()
    main(config)
