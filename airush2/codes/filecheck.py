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

tmp_path = os.path.basename(os.path.normpath(csv_file)).split('_')[0]
print('tmp_path: ', tmp_path)

# dir_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data')
dir_path = '/data/airush2/train/train_data'
label_data_path = os.path.join(DATASET_PATH, 'train',
                                os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')
print(f'label_data_path: {label_data_path}')
file_list= glob.glob(f'{dir_path}/*')
print('file_list: ', file_list)