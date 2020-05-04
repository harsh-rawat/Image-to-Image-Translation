import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, layer_size):
        super().__init__();
        self.relu = nn.ReLU()
        self.layer_1 = nn.Conv2d(layer_size, layer_size, 3, padding=1, stride=1)
        self.layer_1_norm = nn.BatchNorm2d(layer_size)
        self.layer_2 = nn.Conv2d(layer_size, layer_size, 3, padding=1, stride=1)
        self.layer_2_norm = nn.BatchNorm2d(layer_size)

    def forward(self, x):
        out_layer_1 = self.relu(self.layer_1_norm(self.layer_1(x)))
        out_2 = self.layer_2_norm(self.layer_2(out_layer_1)) + x

        return out_2;


class Generator_RESNET(nn.Module):
    def __init__(self, ngf = 64,residual_blocks=9):
        super().__init__();

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # Input is of format batch_size X 1 X 256 X 256
        self.layer_1 = nn.Conv2d(3, ngf, 4, stride=2, padding=1)  # Output would be 64 X 128 X 128
        self.layer_1_norm = nn.BatchNorm2d(ngf)
        self.layer_2 = nn.Conv2d(ngf, ngf*2, 4, stride=2, padding=1)  # Output would be 128 X 64 X 64
        self.layer_2_norm = nn.BatchNorm2d(ngf*2)
        self.layer_3 = nn.Conv2d(ngf*2, ngf*4, 3, stride=1, padding=1)  # Output would be 256 X 64 X 64
        self.layer_3_norm = nn.BatchNorm2d(ngf*4)
        self.layer_4 = nn.Conv2d(ngf*4, ngf*8, 4, stride=2, padding=1)  # Output would be 512 X 32 X 32
        self.layer_4_norm = nn.BatchNorm2d(ngf*8)
        self.layer_5 = nn.Conv2d(ngf*8, ngf*16, 3, stride=1, padding=1)  # Output would be 1024 X 32 X 32
        self.layer_5_norm = nn.BatchNorm2d(ngf*16)

        # Residual Part ###
        self.res_blocks = []
        for i in range(residual_blocks):
            res_block = ResidualBlock(ngf*16);
            if torch.cuda.is_available():
                res_block.cuda()
            self.res_blocks.append(res_block)
        # Residual Part end ###

        self.decode_1 = nn.ConvTranspose2d(ngf*16, ngf*8, 4, stride=2, padding=1)  # Out is 512 X 64 X 64
        self.decode_norm_1 = nn.BatchNorm2d(ngf*8)
        self.decode_2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, stride=2, padding=1)  # Out is 256 X 128 X 128
        self.decode_norm_2 = nn.BatchNorm2d(ngf*4)
        self.decode_3 = nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=1, padding=1)  # Out is 128 X 128 X 128
        self.decode_norm_3 = nn.BatchNorm2d(ngf*2)
        self.decode_4 = nn.ConvTranspose2d(ngf*2, ngf, 3, stride=1, padding=1)  # Out is 64 X 128 X 128
        self.decode_norm_4 = nn.BatchNorm2d(ngf)
        self.decode_5 = nn.ConvTranspose2d(ngf, 3, 4, stride=2, padding=1)  # Out is 3 X 256 X 256

        self._initialize_weights()

    def forward(self, x):

        x = self.relu(self.layer_1_norm(self.layer_1(x)))
        x = self.relu(self.layer_2_norm(self.layer_2(x)))
        x = self.relu(self.layer_3_norm(self.layer_3(x)))
        x = self.relu(self.layer_4_norm(self.layer_4(x)))
        x = self.relu(self.layer_5_norm(self.layer_5(x)))

        for resblock in self.res_blocks:
            x = resblock(x)

        x = self.relu(self.decode_norm_1(self.decode_1(x)))
        x = self.relu(self.decode_norm_2(self.decode_2(x)))
        x = self.relu(self.decode_norm_3(self.decode_3(x)))
        x = self.relu(self.decode_norm_4(self.decode_4(x)))
        x = self.tanh(self.decode_5(x))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, std=0.02)
                nn.init.constant_(m.bias.data, 0)
