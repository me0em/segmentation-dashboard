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
import sys
import logging
import fnmatch
import ntpath
import random

import seaborn as sns

logger = logging.getLogger("dataloader")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def collect_paths(treeroot, pattern):
    results = []
    for base, _, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)

    return sorted(results)

def get_train_test_idxs(paths, coeff=0.95) -> (list, list):
    amount = round(len(paths) * train_subset_coeff)
    idxs = list(range(len(paths)))
    random.shuffle(idxs)
    
    return idxs[:amount], idxs[amount:]

class MinMaxNormalize:
    """Implementation of min-max normalization.
    """
    def __call__(self, sample):
        assert isinstance(sample, (torch.Tensor, np.ndarray)), "arg must be a torch.Tensor or np.ndarray"

        return (sample - sample.min()) / (sample.max() - sample.min())


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    MinMaxNormalize()
])


class XRayRenderSet(Dataset):
    def __init__(self, mask_paths, image_paths, transform):
        self.mask_paths = mask_paths
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.mask_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        original_image = MinMaxNormalize()( transforms.ToTensor()( image ) )
        image = self.transform(image)
        image = image.type(torch.float32)
        
        mask = Image.open(self.mask_paths[idx])
        mask = self.transform(mask)
        mask = mask.type(torch.float32)
        
        return original_image, image, mask


class XRaySet(Dataset):
    def __init__(self, mask_paths, image_paths, transform, name):
        self.mask_paths = mask_paths
        self.image_paths = image_paths
        self.transform = transform
        self.name = name
        
    def __len__(self):
        return len(self.mask_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        image = image.type(torch.float32)
        
        mask = Image.open(self.mask_paths[idx])
        mask = self.transform(mask)
        mask = mask.type(torch.float32)
        
        return image, mask, self.image_paths[idx]
    
def show(x):
    x = einops.rearrange(
        x,
        "b c h w -> (b c h) w"
    )

    x = (x.cpu().detach().numpy() * 255).astype(np.uint8)
    
    return x

def intensity(image, bool_mask, values):
    """ Highlights the masked part of the image
    """
    image = np.stack([image, image, image], axis=2)
    
    if isinstance(values, int):
        values = (values, values, values)

    for ind,v in enumerate(values):
        channel = np.where(~bool_mask, image[:, :, ind], image[:, :, ind] - v)
        channel[channel<0] = 0
        image[:, :, ind] = channel.astype(int)

    return image


def build_the_dataloader(batch_size, img_path, mask_path, name):
    logger.info(f"Try to load data from f{img_path}")

    img_paths = collect_paths(img_path, "*.png")
    mask_paths = collect_paths(mask_path, "*.png")

    # Cut masks with has no corresponding images
    names = [ntpath.split(i)[1] for i in img_paths]
    mask_paths = [i for i in mask_paths if ntpath.split(i)[1] in names]

    # Cut imgs with has no corresponding masks
    names = [ntpath.split(i)[1] for i in mask_paths]
    img_paths = [i for i in img_paths if ntpath.split(i)[1] in names]

    test_idxs = list(range(len(names)))

    logger.info(f"Number of images: {len(img_paths)}\nNumber of masks: {len(mask_paths)}\n")
    logger.info(f"Volume of test dataset: {len(test_idxs)}\n")

    test_image_paths = [img_paths[ind] for ind in test_idxs]
    test_mask_paths  = [mask_paths[ind] for ind in test_idxs]

    test_dataset = XRaySet(image_paths=test_image_paths, mask_paths=test_mask_paths, transform=transform, name=name)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    return test_dataloader
