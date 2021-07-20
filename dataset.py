import os
from glob import glob
import random

from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Lambda

bn_transforms = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor()
            ])    

class datafromcsv(Dataset):
    '''
    Dataset for background blurring
    mode = {train, valid}
    '''
    def __init__(self, root='/home/gt/github/tunet/data/Tune_Dataset/', mode='train', return_name=False, transforms=bn_transforms):

        self.transform = bn_transforms

        self.mode = mode
        self.return_name = return_name
        if self.mode == 'train':
            self.orig_path = f'{root}train_3000_3class/'
            self.blur_path = f'{root}label_3000_3class/'
            self.csv_path  = f'{root}blur_3000_new.csv'
        elif self.mode == 'valid':
            self.orig_path = f'{root}vali_train_600/'
            self.blur_path = f'{root}vali_label_600/'
            self.csv_path  = f'{root}blur_600_new.csv'
        
        self.csvfile = self.csv_extractor(self.csv_path)

    def csv_extractor(self, inputPath_csv):
    
        cols = ["1", "2"]
        csv_df = pd.read_csv(inputPath_csv, sep=",", header=None, names=cols)  
    
        num1s = csv_df["1"].value_counts().keys().tolist()
        num2s = csv_df["1"].value_counts().tolist()

        for (num1, num2) in zip(num1s, num2s):
            if num2 < 2:
                idxs = csv_df[csv_df["1"] == num1].index
                csv_df.drop(idxs, inplace=True)
        return csv_df

    def __len__(self):
        return len(self.csvfile)

    def __getitem__(self, i):
        img_orig = Image.open(f'{self.orig_path}{i}.png').convert('RGB')
        img_orig = self.transform(img_orig)
        img_blur = Image.open(f'{self.blur_path}{i}.png').convert('RGB')
        img_blur = self.transform(img_blur)
        # blur_n = self.csvfile[i].iloc(i)[0]
        blur_n = self.csvfile['1'][i]
        if self.return_name:
            return img_orig, img_blur, blur_n, i
        else:
            return img_orig, img_blur, blur_n