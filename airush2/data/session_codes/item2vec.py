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
from gensim.models import Word2Vec

if not nsml.IS_ON_NSML:
    # if you want to run it on your local machine, then put your path here
    DATASET_PATH = '/home/kwpark_mk2/airush2_temp'
    # print(DATASET_PATH)
    DATASET_NAME = 'airush2_temp'
else:
    from nsml import DATASET_PATH, DATASET_NAME, NSML_NFS_OUTPUT, SESSION_NAME
    print('DATASET_PATH: ', DATASET_PATH)

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


item['read_article_ids'] = item['read_article_ids'].fillna(value='')
article_ids = item['read_article_ids']

article_list = [article.split(',') for article in article_ids if article]

print('Start Item2Vec')
model = Word2Vec(article_list, min_count=5, window=10, size=128, sg=1, negative=5, workers=6, iter=50)
print('End Item2Vec')

print('save item2vec')
model.save('/model/item2vec.model')
print('end item2vec')