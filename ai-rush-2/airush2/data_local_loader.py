import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from PIL import Image
import numpy as np
import pandas as pd
import nsml

from utils import get_transforms
from utils import default_loader

if not nsml.IS_ON_NSML:
    # if you want to run it on your local machine, then put your path here
    DATASET_PATH = '/temp'
    print(DATASET_PATH)
    DATASET_NAME = 'airush2_temp'
else:
    from nsml import DATASET_PATH, DATASET_NAME, NSML_NFS_OUTPUT, SESSION_NAME

    print('DATASET_PATH: ', DATASET_PATH)


class AIRUSH2dataset(Dataset):
    def __init__(self,
                 csv_file,
                 root_dir,
                 args,
                 transform=None,
                 mapping=None,
                 mode='dummy'):

        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            args (argparse object): given arguments of main.py
            transform (callable, optional): Optional transform to be applied
                on a sample.
            mapping (list: (int, int, string), optional): Optional parameter for k-fold
                cross validation.
            mode (string): 'train' or 'valid' or 'test'
        """
        if args['mode']== 'train':
            self.item = pd.read_csv(csv_file,
                                    dtype={
                                        'article_id': str,
                                        'hh': int, 'gender': str,
                                        'age_range': str,
                                        'read_article_ids': str
                                    }, sep='\t')
            label_data_path = os.path.join(DATASET_PATH, 'train',
                                           os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')
            self.label = pd.read_csv(label_data_path,
                                     dtype={'label': int},
                                     sep='\t')

            if nsml.IS_ON_NSML:
                with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'),
                          'rb') as handle:
                    self.image_feature_dict = pickle.load(handle)
            else:
                # on local machine
                with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'),
                          'rb') as handle:
                    self.image_feature_dict = pickle.load(handle)

        else:
            csv_file = os.path.join(csv_file, 'test', 'test_data', 'test_data')

            self.item = pd.read_csv(csv_file,
                                    dtype={
                                        'article_id': str,
                                        'hh': int, 'gender': str,
                                        'age_range': str,
                                        'read_article_ids': str
                                    }, sep='\t')

            if nsml.IS_ON_NSML:
                with open(os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image_features.pkl'),
                          'rb') as handle:
                    self.image_feature_dict = pickle.load(handle)
            else:
                # on local machine
                with open(os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image_features.pkl'),
                          'rb') as handle:
                    self.image_feature_dict = pickle.load(handle)

        self.map = []
        self.mode = mode
        self.args = args
        self.root_dir = root_dir
        self.transform = transform

        if self.args['use_sex']:
            self.sex = {'unknown': 0, 'm': 1, 'f': 2}

        if self.args['use_age']:
            self.age = {'unknown': 0, '-14': 1, '15-19': 2, '20-24': 3, '25-29': 4,
                        '30-34': 5, '35-39': 6, '40-44': 7, '45-49': 8, '50-': 9}

    def __len__(self):
        return len(self.item)

    def __getitem__(self, idx):
        article_id, hh, gender, age_range, read_article_ids = \
            self.item.loc[idx, ['article_id', 'hh', 'gender', 'age_range', 'read_article_ids']]

        if self.args['mode'] == 'train':
            label = self.label.loc[idx, ['label']]
            label = np.array(label, dtype=np.float32)
        else:
            # pseudo label for test mode
            label = np.array(0, dtype=np.float32)

        if nsml.IS_ON_NSML:
            if self.args['mode'] == 'train':
                img_name = os.path.join(self.root_dir, article_id + '.jpg')
            else:  # test, infer
                img_name = os.path.join(self.root_dir, article_id + '.jpg')
        else:
            # on local machine
            if self.args['mode'] == 'train':
                img_name = os.path.join(DATASET_PATH, 'train/train_data/train_image/', article_id + '.jpg')
            else:
                img_name = os.path.join(DATASET_PATH, 'test/test_data/test_image/', article_id + '.jpg')

        image = default_loader(img_name)
        extracted_image_feature = self.image_feature_dict[article_id]

        if self.transform:
            image = self.transform['train'](image)

        # Additional info for feeding FC layer
        flat_features = []
        if self.args['use_sex']:
            sex = self.sex[gender]
            label_onehot = np.zeros(2, dtype=np.float32)
            label_onehot[sex - 1] = 1
            flat_features.extend(label_onehot)

        if self.args['use_age']:
            age = self.age[age_range]
            label_onehot = np.zeros(9, dtype=np.float32)
            label_onehot[age - 1] = 1
            flat_features.extend(label_onehot)

        if self.args['use_exposed_time']:
            time = hh
            label_onehot = np.zeros((24), dtype=np.float32)
            label_onehot[time - 1] = 1
            flat_features.extend(label_onehot)

        if self.args['use_read_history']:
            raise NotImplementedError('If you can handle "sequential" data, then.. hint: this helps a lot')

        flat_features = np.array(flat_features).flatten()
        # pytorch dataloader doesn't accept empty np array
        if flat_features.shape[0] == 0:
            raise NotImplementedError('no flat feature processed. is this on purpose? then delete this line')
            flat_features = np.zeros(1, dtype=np.float32)

        # hint: flat features are concatened into a Tensor, because I wanted to put them all into computational model,
        # hint: if it is not what you wanted, then change the last return line
        return image, extracted_image_feature, label, flat_features


def my_collate(batch):
    from torch.utils.data.dataloader import default_collate
    batch = filter(lambda x: x is not None, batch)
    return default_collate(batch)


def get_data_loader(root, phase, batch_size=16, verbose=True):
    csv_path = root

    data_transforms = get_transforms('[transforms.Resize((456, 232))]', verbose=verbose)
    if phase == 'train':
        print('[debug] data local loader ', phase)
        built_in_args = {'mode': 'train', 'use_sex': True, 'use_age': True, 'use_exposed_time': True,
                         'use_read_history': False,
                         'num_workers': 2, }

        image_datasets = AIRUSH2dataset(
            csv_path,
            os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image'),
            args=built_in_args,
            transform=data_transforms,
            mode='train'
        )
        dataset_sizes = len(image_datasets)

        dataloaders = torch.utils.data.DataLoader(image_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=(built_in_args['mode'] == 'train'),
                                                  pin_memory=False,
                                                  num_workers=built_in_args['num_workers'])

        return dataloaders, dataset_sizes
    elif phase == 'test':
        print('[debug] data local loader ', phase)

        built_in_args = {'mode': 'test', 'use_sex': True, 'use_age': True, 'use_exposed_time': True,
                         'use_read_history': False,
                         'num_workers': 3, }

        image_datasets = AIRUSH2dataset(
            csv_path,
            os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image'),
            args=built_in_args,
            transform=data_transforms,
            mode='test'
        )
        dataset_sizes = len(image_datasets)

        dataloaders = torch.utils.data.DataLoader(image_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=False,
                                                  num_workers=built_in_args['num_workers'])
        return dataloaders, dataset_sizes
    elif phase == 'infer':
        print('[debug] data local loader ', phase)

        built_in_args = {'mode': 'infer', 'use_sex': True, 'use_age': True, 'use_exposed_time': True,
                         'use_read_history': False,
                         'num_workers': 8, }

        image_datasets = AIRUSH2dataset(
            csv_path,
            os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image'),
            args=built_in_args,
            transform=data_transforms,
            mode='test'
        )
        dataset_sizes = len(image_datasets)

        dataloaders = torch.utils.data.DataLoader(image_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=False,
                                                  num_workers=built_in_args['num_workers'])
        return dataloaders, dataset_sizes
    else:
        raise 'mode error'
