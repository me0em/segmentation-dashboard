from einops.layers.torch import Rearrange
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

import numpy as np

import torch

from torch.utils.data import DataLoader, Dataset, random_split

from pl_bolts.models.vision.unet import UNet
from torchvision.transforms import transforms
from .ResNet import ResNet18
from PIL import Image, ImageFilter
from glob import glob


PATH_IMAGES = "../fitting/pure_train"
PATH_MASK = "../fitting/masks_train"

PATH_TEST = "../fitting/test_photos"
PATH_TEST_MASK = "../fitting/test_masks"


class Clasif(LightningModule):
    def __init__(
            self
    ):
        super().__init__()
        # self.save_hyperparameters()

        # networks
        self.model = ResNet18(in_channels=1, outputs=1)


    def forward(self, z):


        return torch.sigmoid(self.model(z))


class Segmenter(LightningModule):
    def __init__(
            self
    ):
        super().__init__()
        # self.save_hyperparameters()


        # networks
        self.model_class = Clasif()#.load_from_checkpoint("../clasif_checkpoint/clasif-1-epoch=21-val_f1=0.9404.ckpt")
        self.model_class.eval()

        self.model = UNet(input_channels=1, num_classes=1, num_layers=5, features_start=64, bilinear=False)

        self.POR = 0.1

        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Resize((1512, 525)),
            # transforms.Normalize((0.5,), (0.24)),
            Rearrange('c (h p1) (w p2) -> (h w) c p1 p2', p1=168, p2=105)
        ])

        self.transform1 = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Resize((1512, 525)),
            # transforms.Normalize((0.5,), (0.24)),
            Rearrange('c (h p1) (w p2) -> (h w) c p1 p2', p1=189, p2=75)
        ])

    def forward(self, z):
        img = self.transform(z)
        img1 = self.transform1(z)
        is_anomalia_pred = self.model_class(img)
        is_anomalia_pred1 = self.model_class(img1)

        flag = is_anomalia_pred.ge(self.POR)
        flag1 = is_anomalia_pred1.ge(self.POR)

        mask_ = self.model(img[flag].unsqueeze(1))
        mask1_ = self.model(img1[flag1].unsqueeze(1))
        mask = torch.zeros(*img.size())
        mask1 = torch.zeros(*img1.size())
        k = 0
        for j, i in enumerate(flag):
            if i == 1:
                mask[j] = mask_[k].detach().cpu()
                k += 1
        k = 0
        for j, i in enumerate(flag1):
            if i == 1:
                mask1[j] = mask1_[k].detach().cpu()
                k += 1

        mask_ = Rearrange('(b1 b2) c h w -> c (b1 h) (b2 w)', b1=9)(mask)
        mask_1 = Rearrange('(b1 b2) c h w -> c (b1 h) (b2 w)', b1=8)(mask1)
        mask_ = torch.max(mask_, mask_1)

        mask_ = mask_.detach().cpu().numpy()

        return mask_


    def predict(self, z):
        z = z[0]
        img = self.transform(z)
        img1 = self.transform1(z)
        is_anomalia_pred = self.model_class(img)
        is_anomalia_pred1 = self.model_class(img1)

        flag = is_anomalia_pred.ge(self.POR)
        flag1 = is_anomalia_pred1.ge(self.POR)

        mask_ = self.model(img[flag].unsqueeze(1))
        mask1_ = self.model(img1[flag1].unsqueeze(1))
        mask = torch.zeros(*img.size())
        mask1 = torch.zeros(*img1.size())
        k = 0
        for j, i in enumerate(flag):
            if i == 1:
                mask[j] = mask_[k].detach().cpu()
                k+=1
        k = 0
        for j, i in enumerate(flag1):
            if i == 1:
                mask1[j] = mask1_[k].detach().cpu()
                k+=1

        mask_ = Rearrange('(b1 b2) c h w -> c (b1 h) (b2 w)', b1=9)(mask)
        mask_1 = Rearrange('(b1 b2) c h w -> c (b1 h) (b2 w)', b1=8)(mask1)
        mask_ = torch.max(mask_, mask_1)

        mask_ = mask_.unsqueeze(1).detach()
        # print(mask_.shape)
        return mask_


def build_the_model(device=None, model_path=None):
    if device is None:
        device = torch.device

    assert model_path is not None, "Need to get model parameters path to load with torch"

    model = Segmenter.load_from_checkpoint(model_path)
    model.eval()
    model = model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Trainable parameters amount: {:n}".format(params))

    # model

    return model


class DatasetTile(Dataset):
    def __init__(self,  path_patch, name, im_prefix, mask_prefix, img_path, mask_path):
        self.im_prefix = im_prefix
        self.mask_prefix = mask_prefix
        self.img_path = img_path
        self.mask_path = mask_path
        self.path_patch = path_patch
        self.name = name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1512, 525)),
            transforms.Normalize((0.5,), (0.24)),
        ])
        self.transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1512, 525)),
        ])

    def __len__(self):
        return len(self.path_patch)

    def __getitem__(self, ind):
        img = Image.open(self.img_path + "/" + self.path_patch[ind] + self.im_prefix)
        mask = Image.open(self.mask_path + "/" + self.path_patch[ind] + self.mask_prefix)

        return self.transform(img), self.transform_mask(mask), self.path_patch[ind]



def build_the_dataloader(batch_size, img_path, mask_path, name):
    im_prefix = ".png"
    mask_prefix = ".png"

    path_im = glob(f"{img_path}/*{im_prefix}")
    path_mask = glob(f"{mask_path}/*{mask_prefix}")
    path_patch = list({i.split(mask_prefix)[0].split("/")[-1] for i in path_mask}.intersection(
        {i.split(im_prefix)[0].split("/")[-1] for i in path_im}))
    # print(path_patch)
    test_dataset = DatasetTile(path_patch, name, im_prefix, mask_prefix, img_path, mask_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return test_dataloader



# build_the_model(model_path="./segmenter_checkpoint/seg-1-epoch=10-val_iou=0.3343.ckpt")



