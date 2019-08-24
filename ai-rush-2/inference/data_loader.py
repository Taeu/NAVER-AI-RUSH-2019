import os
import json
import torch
import pickle
import logging
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from utils import *

import nsml
from nsml import DATASET_PATH, DATASET_NAME


def feed_infer(output_file, infer_func):
    """
    output_file (str): file path to write output (Be sure to write in this location.)
    infer_func (function): user's infer function bound to 'nsml.bind()'
    """
    try:
        import nsml
        root = os.path.join(nsml.DATASET_PATH)
        print('[debug][data_loader] root path : ', root)
    except:
        # this is for local debug use. not for nsml leader board
        raise AssertionError('this is for local debug use. not for nsml leader board.')

    predicted_labels = infer_func(root, phase='test')
    predicted_labels = ' '.join([str(label) for label in predicted_labels])

    with open(output_file, 'w') as f:
        f.write(predicted_labels)
    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')
