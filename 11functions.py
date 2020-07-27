import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.style.use("seaborn-darkgrid")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnext50_32x4d
from torch.optim import Adam, AdamW, lr_scheduler

from tqdm import trange
import os
import albumentations as albu
import cv2
import glob

def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def dice_loss(inputs, target):
    num = target.size(0)
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice

def bce_dice_loss(inputs, target):
    dicescore = dice_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore

def train_one_epoch(model, optimizer, lr_scheduler, data_loader, epoch):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Start Train ...")
    model.train()

    losses = []
    accur = []

    for data, target in data_loader:
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


        data = data.permute(0,3,1,2).to(device)
        targets = target.permute(0,3,1,2).to(device)

        outputs = model(data)

        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

        train_dice = dice_metric(out_cut, targets.data.cpu().numpy())

        loss = bce_dice_loss(outputs, targets)

        losses.append(loss.item())
        accur.append(train_dice)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if lr_scheduler is not None:
        lr_scheduler.step()

    print("Epoch [%d]" % (epoch))
    print("Mean loss on train:", np.array(losses).mean(), "Mean DICE on train:", np.array(accur).mean())

    return np.array(losses).mean(), np.array(accur).mean()

def val_epoch(model, data_loader_val, epoch, threshold=0.33):
  
    if epoch is None:
        print("Start Test...")
    else:
        print("Start Validation ...")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    val_acc = []

    with torch.no_grad():
        for data, targets in data_loader_val:
            
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            data = data.permute(0,3,1,2).to(device)
            targets = targets.permute(0,3,1,2).to(device)

            outputs = model(data)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            val_dice = dice_metric(out_cut, targets.data.cpu().numpy())
            val_acc.append(val_dice)

        print("Epoch:  " + str(epoch) + "  Threshold:  " + str(threshold)\
              + " Mean DICE on validation:", np.array(val_acc).mean())
        
        return np.array(val_acc).mean()
    
    
    
def plot_history(train_history,
                       val_history,
                       loss_history ,
                       num_epochs):
    
    x = np.arange(num_epochs)

    fig = plt.figure(figsize=(16, 6))
    plt.plot(x, train_history, label='train dice', lw=3, c="green")
    plt.plot(x, val_history, label='validation dice', lw=3, c="red")
    plt.plot(x, loss_history, label='dice + bce', lw=3)

    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("DICE", fontsize=15)
    plt.legend()

    return plt.show()
