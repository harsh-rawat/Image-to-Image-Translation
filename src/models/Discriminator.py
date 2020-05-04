import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_size=256, leaky_relu=0.2):

        super(Discriminator, self).__init__()

        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        self.sigmoid = nn.Sigmoid()

        # Considering input to be 256
        self.layer_1 = nn.Conv2d(3, 32, 4, padding=1, stride=2)
        self.layer_1_bn = nn.BatchNorm2d(32)
        # Output size is batch_size X 32 X 128 X 128

        self.layer_2 = nn.Conv2d(32, 64, 4, padding=1, stride=2)
        self.layer_2_bn = nn.BatchNorm2d(64)
        # Output size is batch_size X 64 X 64 X 64

        self.layer_3 = nn.Conv2d(64, 128, 4, padding=1, stride=2)
        self.layer_3_bn = nn.BatchNorm2d(128)
        # Output size is batch_size X 128 X 32 X 32

        self.layer_4 = nn.Conv2d(128, 256, 4, padding=1, stride=2)
        self.layer_4_bn = nn.BatchNorm2d(256)
        # Output size is batch_size X 256 X 16 X 16

        self.layer_5 = nn.Conv2d(256, 512, 4, padding=1, stride=2)
        self.layer_5_bn = nn.BatchNorm2d(512)
        # Output size is batch_size X 512 X 8 X 8

        self.layer_6 = nn.Conv2d(512, 1, int(image_size / 32), padding=0, stride=1)

        self._initialize_weights()

    def forward(self, x):
        # Considering x to be of shape (batch_size X 3 X image_size X image_size)
        x = self.leaky_relu(self.layer_1_bn(self.layer_1(x)))

        x = self.leaky_relu(self.layer_2_bn(self.layer_2(x)))

        x = self.leaky_relu(self.layer_3_bn(self.layer_3(x)))

        x = self.leaky_relu(self.layer_4_bn(self.layer_4(x)))

        x = self.leaky_relu(self.layer_5_bn(self.layer_5(x)))

        x = self.sigmoid(self.layer_6(x))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, std=0.02)
                nn.init.constant_(m.bias.data, 0)
