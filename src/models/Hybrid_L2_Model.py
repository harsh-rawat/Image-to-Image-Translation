import torch.nn as nn
from torchvision.utils import save_image
import torch
import torchvision.models as models

from models.Model import Model


class Hybrid_L2_Model(Model):
    def __init__(self, base_path='', epochs=10, learning_rate=0.0002, image_size=256, leaky_relu=0.2,
                 betas=(0.5, 0.999), lamda=100, image_format='png'):
        super().__init__(base_path, epochs, learning_rate, image_size, leaky_relu, betas, lamda, image_format)
        print('We will be using L2 loss with perceptual loss!')

    def calculate_image_similarity_loss(self, img1, img2):
        l1_loss = nn.L1Loss()
        l2_loss = nn.MSELoss()
        gen_pre_train = l1_loss(self.vgg16_conv(img1), self.vgg16_conv(img2))
        similarity_loss = l2_loss(img1, img2) + gen_pre_train
        return similarity_loss
