import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnext50_32x4d
from torch.optim import Adam, AdamW, lr_scheduler
import numpy as np
from tqdm import trange
import os
import albumentations as albu
import cv2
import glob

class BrainMRI(Dataset):
    def __init__(self, data, transforms, n_classes=3):
        
        self.data = data
        self.transforms = transforms
        self.n_classes = n_classes
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        image = self.data[idx][0].astype("float32")

        # standardize data
        mean, std = image.mean(), image.std()
        image = (image - mean) / std
        
        # convert to rgb
        image_rgb = np.stack([image]*3).transpose(1,2,0)
        
        # create target masks
        label = self.data[idx][2] -1
        mask = np.expand_dims(self.data[idx][1], -1)
        
        target_mask = np.zeros((mask.shape[0], mask.shape[1], 
                                self.n_classes))
        target_mask[:,:, label : label + 1] = mask.astype("uint8")
        
        #  clip to binary mask
        target_mask = np.clip(target_mask, 0, 1).astype("float32")
        
        # augmentations
        augmented_image_mask = self.transforms(image=image_rgb, 
                                    mask=target_mask)
        image = augmented_image_mask['image']
        mask = augmented_image_mask['mask']
        
        return image, mask
class ResNeXtUNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        
        # Down
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        # Up
        self.decoder4 = DecoderBlock(2048, 1024)
        self.decoder3 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder1 = DecoderBlock(256, 256)

        # Final Classifier
        self.last_conv0 = ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)
                       
        
    def forward(self, x):
        # Down
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Up + sc
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        

        # Final classifier
        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = torch.sigmoid(out)
        
        return out
    
    
class ConvRelu(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()

        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convrelu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)
        
        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        
        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)

        return x
