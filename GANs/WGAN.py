import torch
import torch.nn as nn

# Define critic
class Critic(nn.Module):
    def __init__(self, channels, features_d):
        super().__init__()
        self.dis = nn.Sequential(
            # Input: N x channels x 64 x 64
            nn.Conv2d(
                channels, features_d, kernel_size=4, stride=2, padding=1
            ),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_c, out_c, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.dis(x)

# Define generator
class Generator(nn.Module):
    def __init__(self, z_dim, image_dim, features_g):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 2, 0), # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1), #8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1), #16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1), #32x32
            nn.ConvTranspose2d(features_g * 2, image_dim, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_c, out_c, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_c, 
                out_c, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.gen(x)


# Init model with normal distribution with zero mean and 0.02 std
def init_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)


# Test generator and discriminator
def test():
    # Image example
    N, channels, H, W = 8, 3, 64, 64
    Z_dim = 100
    x = torch.randn(N, channels, H, W)
    disc = Critic(channels, 8)
    init_model(disc)
    # Discriminator should give just probability true image
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(Z_dim, channels, 8)
    z = torch.randn((N, Z_dim, 1, 1))
    # Generator should give image of train shape
    assert gen(z).shape == (N, channels, H, W)

    print("Success!")