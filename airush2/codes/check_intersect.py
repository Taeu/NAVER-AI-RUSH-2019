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

item_article_list = item['article_id'].tolist()
article_id_list = article_df['article_id'].tolist()

inter_set = set(item_article_list) & set(article_id_list)
print(f'item set: {len(set(item_article_list))}')
print(f'article set: {len(set(article_id_list))}')
print(f'inter_set len: {len(inter_set)}')

# label_data_path = os.path.join(DATASET_PATH, 'train',
#                                 os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')
# label = pd.read_csv(label_data_path,
#                     dtype={'label': int},
#                     sep='\t')


# with open(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl'), 'rb') as handle:
#     image_feature_dict = pickle.load(handle)

# article_ids = item['article_id'].tolist()[:100]

# for article_id in article_ids[:50]:
#     print(article_id)
#     for elm in image_feature_dict[article_id]:
#         print(elm)
#     print('='*50)