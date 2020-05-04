"""
@author: harsh
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import pathlib
from PIL import Image
import re


def load_image(file_path):
    with open(file_path, 'rb') as file:
        with Image.open(file) as img:
            return img.convert('RGB')


class CustomDataset(Dataset):
    def __init__(self, path, image_size, image_format='png', image_type='rgb'):
        self.root = path
        self.image_size = image_size
        self.image_type = image_type

        path_loc = pathlib.Path(path)
        if not path_loc.exists():
            raise Exception('The path provided is incorrect!')

        searchstring = os.path.join(path, '*.' + image_format)
        list_of_images = glob.glob(searchstring)
        list_of_images.sort(key=self.natural_keys)
        self.image_paths = list_of_images

    def __getitem__(self, index):
        file_path = self.image_paths[index]
        img = load_image(file_path)
        img = img.resize((self.image_size, self.image_size))
        img_np = np.array(img)

        # Scale the values to range -1 to 1
        img_np = (img_np - 127.5) / 127.5
        img_np = np.transpose(img_np, (2, 0, 1))

        return torch.FloatTensor(img_np)

    def atof(self, text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    def natural_keys(self, text):
        return [self.atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

    def __len__(self):
        return len(self.image_paths)

    def __str__(self):
        return 'Dataset details - \nRoot Location : {}\nImage Size : {}\nSize : {}'.format(
            self.root, self.image_size, self.__len__())
