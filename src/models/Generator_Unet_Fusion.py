import torch.nn as nn
import torchvision.models as models
import torch


class Generator_Unet_Fusion(nn.Module):
    def __init__(self, image_size=256, ngf=64):
        super(Generator_Unet_Fusion, self).__init__()

        self.inception2 = models.googlenet(pretrained=True, progress=True)
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        self.layer_0 = nn.Conv2d(3, ngf, 3, padding=1, stride=1)  # 256 X 256
        self.layer_0_bn = nn.BatchNorm2d(ngf)

        self.layer_1 = nn.Conv2d(ngf, ngf, 4, padding=1, stride=2)  # 128 X 128
        self.layer_1_bn = nn.BatchNorm2d(ngf)

        self.layer_2 = nn.Conv2d(ngf, ngf * 2, 4, padding=1, stride=2)  # 64 X 64
        self.layer_2_bn = nn.BatchNorm2d(ngf * 2)

        self.layer_3 = nn.Conv2d(ngf * 2, ngf * 4, 4, padding=1, stride=2)  # 32 X 32
        self.layer_3_bn = nn.BatchNorm2d(ngf * 4)

        self.layer_4 = nn.Conv2d(ngf * 4, ngf * 8, 4, padding=1, stride=2)  # 16 X 16
        self.layer_4_bn = nn.BatchNorm2d(ngf * 8)

        self.layer_5 = nn.Conv2d(ngf * 8, ngf * 8, 4, padding=1, stride=2)  # 8 X 8
        self.layer_5_bn = nn.BatchNorm2d(ngf * 8)

        self.layer_6 = nn.Conv2d(ngf * 8, ngf * 8, 4, padding=1, stride=2)  # 4 X 4
        self.layer_6_bn = nn.BatchNorm2d(ngf * 8)

        self.layer_7 = nn.Conv2d(ngf * 8, ngf * 8, 4, padding=1, stride=2)  # 2 X 2
        self.layer_7_bn = nn.BatchNorm2d(ngf * 8)

        # Adding fusion layer here

        self.layer_decode_6 = nn.ConvTranspose2d(ngf * 8 + 1000, ngf * 8, 4, padding=1, stride=2)  # 4X4
        self.layer_decode_6_bn = nn.BatchNorm2d(ngf * 8)

        self.layer_decode_5 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, padding=1, stride=2)  # 8X8
        self.layer_decode_5_bn = nn.BatchNorm2d(ngf * 8)

        self.layer_decode_4 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, padding=1, stride=2)  # 16X16
        self.layer_decode_4_bn = nn.BatchNorm2d(ngf * 8)

        self.layer_decode_3 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, padding=1, stride=2)  # 32X32
        self.layer_decode_3_bn = nn.BatchNorm2d(ngf * 4)

        self.layer_decode_2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, padding=1, stride=2)  # 64X64
        self.layer_decode_2_bn = nn.BatchNorm2d(ngf * 2)

        self.layer_decode_1 = nn.ConvTranspose2d(ngf * 2, ngf, 4, padding=1, stride=2)  # 128X128
        self.layer_decode_1_bn = nn.BatchNorm2d(ngf)

        self.layer_decode_0 = nn.ConvTranspose2d(ngf, ngf, 4, padding=1, stride=2)  # 256X256
        self.layer_decode_0_bn = nn.BatchNorm2d(ngf)

        self.output = nn.Conv2d(ngf, 3, 3, padding=1, stride=1)  # 256 X 256

        self._initialize_weights()

    def forward(self, x):

        rgb = (x+1)/2
        inception_features = self.inception2(rgb)

        x = self.leaky_relu(self.layer_0_bn(self.layer_0(x)))
        store_0 = x

        x = self.leaky_relu(self.layer_1_bn(self.layer_1(x)))
        store_1 = x

        x = self.leaky_relu(self.layer_2_bn(self.layer_2(x)))
        store_2 = x

        x = self.leaky_relu(self.layer_3_bn(self.layer_3(x)))
        store_3 = x

        x = self.leaky_relu(self.layer_4_bn(self.layer_4(x)))
        store_4 = x

        x = self.leaky_relu(self.layer_5_bn(self.layer_5(x)))
        store_5 = x

        x = self.leaky_relu(self.layer_6_bn(self.layer_6(x)))
        store_6 = x

        x = self.leaky_relu(self.layer_7_bn(self.layer_7(x)))
        x = self.fusion(x, inception_features)

        x = self.layer_decode_6(x) + store_6
        x = self.relu(self.layer_decode_6_bn(x))

        x = self.layer_decode_5(x) + store_5
        x = self.relu(self.layer_decode_5_bn(x))

        x = self.layer_decode_4(x) + store_4
        x = self.relu(self.layer_decode_4_bn(x))

        x = self.layer_decode_3(x) + store_3
        x = self.relu(self.layer_decode_3_bn(x))

        x = self.layer_decode_2(x) + store_2
        x = self.relu(self.layer_decode_2_bn(x))

        x = self.layer_decode_1(x) + store_1
        x = self.relu(self.layer_decode_1_bn(x))

        x = self.layer_decode_0(x) + store_0
        x = self.relu(self.layer_decode_0_bn(x))

        x = self.tanh(self.output(x))

        return x

    def fusion(self, x, features):
        size = x.shape
        features = features.repeat(size[2], size[3], 1, 1)
        features = features.permute(2, 3, 0, 1)

        combined = torch.cat((x, features), 1)
        return combined

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
