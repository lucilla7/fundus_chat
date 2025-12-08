import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class FundusSegDataset(Dataset):
    def __init__(self, root, size=512, augment=True):
        self.root = root
        self.size = size
        self.augment = augment
        
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")

        self.files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        ip = os.path.join(self.img_dir, fname)
        mp = os.path.join(self.mask_dir, fname)

        img = cv2.imread(ip)
        mask = cv2.imread(mp, 0)

        # resize
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))

        # normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # augmentations
        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            if random.random() < 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)

        # HWC → CHW
        img = img.transpose(2,0,1)

        return torch.tensor(img), torch.tensor(mask).unsqueeze(0)
    
    