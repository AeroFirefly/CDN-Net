from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch
import cv2
from astropy.io import fits
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from matplotlib import pyplot as plt
import sep
from .utils import seed_torch

# seed_torch()

class TrainSetLoaderFits(Dataset):
    
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, preprocess='hierarchy', mask_type='soft',
                 base_size=256, crop_size=256, transform=None, suffix='.png', label_mode='seg'):
        super(TrainSetLoaderFits, self).__init__()

        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.fits = dataset_dir+'/'+'fits'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        self.label_mode = label_mode
        self.preprocess = preprocess.lower()
        self.mask_type = mask_type.lower()

    def _preprocess(self, fits_img, mask, preprocess, bkg):
        # preprocess method
        img = np.expand_dims(np.empty(fits_img.shape, dtype='uint8'), 2).repeat(3, axis=2) if preprocess=='hierarchy' \
                else fits_img[:, :, np.newaxis]
        
        if preprocess=='hierarchy':
            fits_img = fits_img.astype(dtype='uint16')
            img[:, :, 0] = (fits_img >> 8) & 0xff
            img[:, :, 1] = (fits_img >> 4) & 0xff
            img[:, :, 2] = fits_img & 0xff
        else:
            raise Exception('wrong preprocess method')

        # data enhancement
        num_ch = img.shape[-1]  # channels account, 1 or 3
        if random.random() < 0.5:   # random mirror
            for i in range(num_ch):
                img[:, :, i] = img[:, ::-1, i]
                mask = mask[:, ::-1]
        if random.random() < 0.5:   # gaussian blur as in PSP
            sigma = random.random()
            ksize = 3
            for i in range(num_ch):
                img[:, :, i] = cv2.GaussianBlur(img[:, :, i], (ksize, ksize), sigma)

        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        
        return img, mask

    def __getitem__(self, idx):
        img_id     = self._items[idx]                           
        img_path   = self.fits + '/' + img_id + '.fits'           
        label_path = self.masks +'/' + img_id + self.suffix
        preprocess = self.preprocess
        mask_type = self.mask_type

        # load fits
        fits_img = fits.getdata(img_path)  # uint16
        bkg = sep.Background(fits_img.astype(np.float32))
        fits_img = fits_img - bkg
        fits_img[fits_img < 0] = 0

        # load masks
        mask = Image.open(label_path)
        mask = np.array(mask, dtype=np.float32)

        # preprocess method
        img, mask = self._preprocess(fits_img, mask, preprocess, bkg)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        return img, torch.from_numpy(mask) #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoaderFits(Dataset):
    
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, preprocess='hierarchy', 
                 base_size=256, crop_size=256, transform=None, suffix='.png', label_mode='seg'):
        super(TestSetLoaderFits, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.ipds     = dataset_dir + '/' + 'masks'
        self.fits = dataset_dir + '/' + 'fits'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix
        self.label_mode = label_mode
        self.preprocess = preprocess.lower()

    def _testval_preprocess(self, fits_img, preprocess, bkg):
        # preprocess method
        img = np.expand_dims(np.empty(fits_img.shape, dtype='uint8'), 2).repeat(3, axis=2) if preprocess=='hierarchy' \
                else fits_img[:, :, np.newaxis]
        
        if preprocess=='hierarchy':
            fits_img = fits_img.astype(dtype='uint16')
            img[:, :, 0] = (fits_img >> 8) & 0xff
            img[:, :, 1] = (fits_img >> 4) & 0xff
            img[:, :, 2] = fits_img & 0xff
        else:
            raise Exception('wrong preprocess method')
        
        return img

    def __getitem__(self, idx):
        img_id = self._items[idx]  
        img_path   = self.fits + '/' + img_id + '.fits'
        preprocess = self.preprocess

        # load fits
        fits_img = fits.getdata(img_path)  # uint16
        bkg = sep.Background(fits_img.astype(np.float32))
        fits_img = fits_img - bkg
        fits_img[fits_img < 0] = 0

        # preprocess method
        img = self._testval_preprocess(fits_img, preprocess, bkg)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, img_id

    def __len__(self):
        return len(self._items)