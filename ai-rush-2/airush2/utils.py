import os
import pickle
import logging
import asyncio
from io import BytesIO
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image, ImageOps

import nsml

from torchvision import utils, datasets, transforms

from nsml import DATASET_PATH, DATASET_NAME


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # https://github.com/python-pillow/Pillow/issues/835
    with open(path, 'rb') as f:
        img = Image.open(f)
        ##todo: hardcoded to handle "image file is truncated" problem

        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_transforms(tf_args='[transforms.Resize((456, 232))]', verbose=True):
    if tf_args != 'Preprocessed':
        if verbose:
            print("tf args", tf_args)
        train_tf = eval(tf_args)
        train_tf = train_tf + [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    data_transforms = {
        'train': transforms.Compose(train_tf)
    }
    return data_transforms
