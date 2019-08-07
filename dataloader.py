import os
import numpy as np
import pathlib
import pandas as pd
import pickle as pkl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nsml import DATASET_PATH

def train_dataloader(input_size=128,
                    batch_size=64,
                    num_workers=0,
                    ):
    
    image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images') 
    train_label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
    train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
    train_meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)
        
    dataloader = DataLoader(
        AIRushDataset(image_dir, train_meta_data, label_path=train_label_path, 
                      transform=transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    return dataloader


class AIRushDataset(Dataset):
    def __init__(self, image_data_path, meta_data, label_path=None, transform=None):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_path = label_path
        self.transform = transform
        
        if self.label_path is not None:
            self.label_matrix = np.load(label_path)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir , str(self.meta_data['package_id'].iloc[idx]) , str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load() # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3]) # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)
        
        if self.label_path is not None:
            tags = torch.tensor(np.argmax(self.label_matrix[idx])) # here, we will use only one label among multiple labels.
            return new_img, tags
        else:
            return new_img
