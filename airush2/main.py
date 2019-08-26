import glob
import pickle
import pandas as pd
from data_local_loader import get_data_loader
import os
import argparse
import numpy as np
import time
import datetime

import nsml

from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import get_transforms
from utils import default_loader

from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

if not nsml.IS_ON_NSML:
    # if you want to run it on your local machine, then put your path here
    DATASET_PATH = '/home/kwpark_mk2/airush2_temp'
    # print(DATASET_PATH)
    DATASET_NAME = 'airush2_temp'
else:
    from nsml import DATASET_PATH, DATASET_NAME, NSML_NFS_OUTPUT, SESSION_NAME
    print('DATASET_PATH: ', DATASET_PATH)

csv_file = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data')
article_file = os.path.join(DATASET_PATH, 'train', 
                            'train_data', 'train_data_article.tsv')

article_df = pd.read_csv(article_file, sep='\t')

item = pd.read_csv(csv_file,
                    dtype={
                        'article_id': str,
                        'hh': int, 'gender': str,
                        'age_range': str,
                        'read_article_ids': str
                    }, sep='\t')

item = item.fillna('')

label_data_path = os.path.join(DATASET_PATH, 'train',
                                os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')
label = pd.read_csv(label_data_path,
                    dtype={'label': int},
                    sep='\t')

X = item.values
y = label['label'].values

train_x, train_y = RandomUnderSampler(random_state=42).fit_sample(X, y)

print(train_y.shape)
print(np.unique(train_y, return_counts=True))

# def create_readlist(row):
#     if row:
#         return row.split(',')
#     else:
#         return []

# item = item.fillna('')  # fillna
# item['read_article_list'] = item['read_article_ids'].apply(create_readlist)
# item['read_len'] = item['read_article_list'].apply(len)

# print(item['read_len'].describe())



# print('label.value_counts(): ', label['label'].value_counts())


# with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'), 'rb') as handle:
#     image_feature_dict = pickle.load(handle)

# article_ids = item['article_id'].tolist()[:100]

# for article_id in article_ids[:50]:
#     print(article_id)
#     for elm in image_feature_dict[article_id]:
#         print(elm)
#     print('='*50)