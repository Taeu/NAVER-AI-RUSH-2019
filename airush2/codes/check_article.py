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

print(f'article_len: {len(article_df)}')
print(article_df.head())

print('='*50)
print('check unique category')
print(article_df['category_id'].value_counts())

print('='*50)
print('article_id')
for article_id in article_df['article_id'].tolist()[:100]:
    print(article_id)

print('='*50)
print('category_id')
for cat_id in article_df['category_id'].tolist()[:100]:
    print(cat_id)

print('='*50)
print('title')
for t in article_df['title'].tolist()[:100]:
    print(t)