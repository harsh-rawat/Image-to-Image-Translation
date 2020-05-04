import torch.nn as nn


class Generator_Unet_Shallow(nn.Module):
    def __init__(self, image_size=256, ngf=64):
        super(Generator_Unet_Shallow, self).__init__()

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.layer_1 = nn.Conv2d(3, ngf, 4, padding=1, stride=2)
        self.layer_1_bn = nn.BatchNorm2d(ngf)

        self.layer_2 = nn.Conv2d(ngf, ngf*2, 4, padding=1, stride=2)
        self.layer_2_bn = nn.BatchNorm2d(ngf*2)

        self.layer_3 = nn.Conv2d(ngf*2, ngf*4, 4, padding=1, stride=2)
        self.layer_3_bn = nn.BatchNorm2d(ngf*4)

        self.layer_4 = nn.Conv2d(ngf*4, ngf*8, 4, padding=1, stride=2)
        self.layer_4_bn = nn.BatchNorm2d(ngf*8)

        self.layer_5 = nn.Conv2d(ngf*8, ngf*8, 4, padding=1, stride=2)
        self.layer_5_bn = nn.BatchNorm2d(ngf*8)

        self.layer_6 = nn.ConvTranspose2d(ngf*8, ngf*8, 4, padding=1, stride=2)
        self.layer_6_bn = nn.BatchNorm2d(ngf*8)

        self.layer_7 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, padding=1, stride=2)
        self.layer_7_bn = nn.BatchNorm2d(ngf*4)

        self.layer_8 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, padding=1, stride=2)
        self.layer_8_bn = nn.BatchNorm2d(ngf*2)

        self.layer_9 = nn.ConvTranspose2d(ngf*2, ngf, 4, padding=1, stride=2)
        self.layer_9_bn = nn.BatchNorm2d(ngf)

        self.layer_10 = nn.ConvTranspose2d(ngf, 3, 4, padding=1, stride=2)

        self._initialize_weights()

    def forward(self, x):

        x = self.relu(self.layer_1_bn(self.layer_1(x)))
        store_1 = x

        x = self.relu(self.layer_2_bn(self.layer_2(x)))
        store_2 = x

        x = self.relu(self.layer_3_bn(self.layer_3(x)))
        store_3 = x

        x = self.relu(self.layer_4_bn(self.layer_4(x)))
        store_4 = x

        x = self.relu(self.layer_5_bn(self.layer_5(x)))

        x = self.relu(self.layer_6_bn(self.layer_6(x)))
        x += store_4

        x = self.relu(self.layer_7_bn(self.layer_7(x)))
        x += store_3

        x = self.relu(self.layer_8_bn(self.layer_8(x)))
        x += store_2

        x = self.relu(self.layer_9_bn(self.layer_9(x)))
        x += store_1

        x = self.tanh(self.layer_10(x))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, std=0.02)
                nn.init.constant_(m.bias.data, 0)
