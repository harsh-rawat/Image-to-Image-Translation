import torch
from data.CustomDataset import CustomDataset
from random import shuffle
import pickle
import pathlib
import os


class Dataloader:
    def __init__(self, path, image_size, batch_size=16, image_format='png', train=True):
        if train:
            path = os.path.join(path, 'train')
        else:
            path = os.path.join(path, 'test')

        self.path = path
        self.image_size = image_size
        self.image_format = image_format
        self.batch_size = batch_size

    def get_data_loader(self, sub_dir=None):
        if sub_dir is None:
            sub_dir = os.listdir(self.path)
            if len(sub_dir) < 2:
                raise Exception('Incorrect Data path or the data is not in correct format!')

        x_path = os.path.join(self.path, sub_dir[0])
        y_path = os.path.join(self.path, sub_dir[1])

        print('X is {} and Y is {}'.format(sub_dir[0], sub_dir[1]))

        x_dataset = CustomDataset(x_path, self.image_size, self.image_format)
        y_dataset = CustomDataset(y_path, self.image_size, self.image_format)

        x_loader = torch.utils.data.DataLoader(x_dataset, shuffle=True, batch_size=self.batch_size)
        y_loader = torch.utils.data.DataLoader(y_dataset, shuffle=True, batch_size=self.batch_size)

        return x_loader, y_loader
