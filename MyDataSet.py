# -*- coding: utf-8 -*-
import glob
import os
from skimage import io
from PIL import Image

import torch
from torch.utils.data import Dataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class JDDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, cats=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        filenames = []
        if cats is None:
            cats = [os.path.join(root_dir, o) for o in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir,o))]
            filenames = glob.glob(os.path.join(root_dir, "*/*.jpg"))
        else:
            for cat in cats:
                for fn in glob.glob(os.path.join(root_dir, cat +"/"+ "*.jpg")):
                    filenames.append(fn)

        self.filenames = filenames
        self.cats = [os.path.split(os.path.split(filename)[0])[-1] for filename in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = pil_loader(self.filenames[idx])
        if self.transform:
            image = self.transform(image)

        return (image, self.cats[idx])