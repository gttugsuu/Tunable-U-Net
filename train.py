import os
import argparse
from tqdm import tqdm
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tunet import Tunet
from dataset import datafromcsv
from utility import VisdomLinePlotter

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
print('Using ', device)

output_path = 'tunet-blur/'

# Dataset, dataloader
params = {
    'batch_size': 10,
    'shuffle': True,
    'num_workers': 8,
    'drop_last': False
}
train_loader = DataLoader(datafromcsv(mode='train', ), **params)
valid_loader = DataLoader(datafromcsv(mode='valid', ), **params)

# Initialize model
model = Tunet(n_channels=3).to(device)

# Loss functions
maeloss = nn.L1Loss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Logger
visdom_plotter = VisdomLinePlotter(env_name=f'tunet-blur/', filepath=f'{output_path}log.visdom')

for epoch in range(1000):
    print(f'Epoch: {epoch}')

    model.train()
    trainloss = 0.0
    for img_orig, img_bn, bn_n in tqdm(train_loader, desc='train'):
        img_orig = img_orig.to(device)
        img_bn = img_bn.to(device)
        bn_n = bn_n.unsqueeze(-1).float().to(device)

        output = model(img_orig, bn_n)

        loss = maeloss(output, img_bn)
        trainloss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    validloss = 0.0
    for img_orig, img_bn, bn_n in tqdm(valid_loader, desc='valid'):
        img_orig = img_orig.to(device)
        img_bn = img_bn.to(device)
        bn_n = bn_n.unsqueeze(-1).float().to(device)

        output = model(img_orig, bn_n)

        loss = maeloss(output, img_bn)
        validloss += loss.item()

    visdom_plotter.plot(legend_name='train', title_name='Train loss', x=epoch, y=trainloss/len(train_loader))
    visdom_plotter.plot(legend_name='valid', title_name='Train loss', x=epoch, y=validloss/len(valid_loader))

    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, f'{output_path}{epoch}.pth')

    scheduler.step()