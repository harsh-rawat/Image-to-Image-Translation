import torch.nn as nn
from models.Model import Model


class L2_Model(Model):
    def __init__(self, base_path='', epochs=10, learning_rate=0.0002, image_size=256, leaky_relu=0.2,
                 betas=(0.5, 0.999), lamda=100, image_format='png'):
        super().__init__(base_path, epochs, learning_rate, image_size, leaky_relu, betas, lamda, image_format)
        print('We will be using L2 loss only!')

    def calculate_image_similarity_loss(self, img1, img2):
        l2_loss = nn.MSELoss()
        similarity_loss = l2_loss(img1, img2)
        return similarity_loss
