import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from self_attention_cv.transunet import TransUnet

import numpy as np
import einops

import matplotlib.pyplot as plt
from PIL import Image

import os
import fnmatch
import ntpath
import random

import seaborn as sns


class MyTransUNet(TransUnet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        x = super().forward(x)
        x = torch.sigmoid(x)
        
        return x
    
    def predict(self, x):
        if torch.cuda.is_available():
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        with torch.no_grad():
            x = self.forward(x)
            threshold = torch.quantile(x.flatten(), torch.tensor([0.01]).to(device))
            x = (x > threshold).type(torch.float32)
            # x = x.type(torch.float32)  # without the threshold
            
        return x


def build_the_model(device=None, model_path=None):
    if device is None:
        device = torch.device
    
    assert model_path is not None, "Need to get model parameters path to load with torch"
    
    model = MyTransUNet(
        in_channels=1,
        img_dim=256,
        vit_blocks=8,  # было 8
        vit_dim_linear_mhsa_block=512,  #  лучшее 512
        classes=1
    )
    
    model = model.to(device)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Trainable parameters amount: {:n}".format(params))
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model