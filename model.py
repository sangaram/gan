import torch
from torch import nn
from torch.nn import functional as F


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        
    def forward(self, x:torch.Tensor):
        out = self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))
        out = out + x
        
        return out
    
class Generator(nn.Module):
    """Generator network for the GAN

    Attributes
    ----------
    residual_blocks : int
        The number of residual blocks

    channels : int
        The number of channels of the input images. Either 3 or 1.
    """
    def __init__(self, residual_blocks:int, channels:int=3):
        super().__init__()
        self.residual_blocks = residual_blocks
        self.channels = channels
        self.in_conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.blocks = nn.Sequential(*[
            GeneratorBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ) for _ in range(residual_blocks)
        ])
        
        self.out_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        self.out_block2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.out_block3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.out_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        
        
    def forward(self, x:torch.Tensor):
        x = self.in_conv(x)
        x = self.out_block1(self.blocks(x)) + x
        x = self.out_conv(self.out_block3(self.out_block2(x)))
        return x
    

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU()
    
    def forward(self, x:torch.Tensor):
        return self.lrelu(self.bn(self.conv(x)))
    
class Discriminator(nn.Module):
    """The discriminator network of the GAN

    Attributes
    ----------
    channels : int
        The number of channels of the input images. Either 3 or 1.
    """
    def __init__(self, channels:int=3):
        super().__init__()
        self.channels = channels
        self.in_conv = nn.Conv2d(channels, 64, kernel_size=3)
        self.lrelu = nn.LeakyReLU(0.2)
        self.blocks = nn.Sequential(
            # shape (*, 64, 382, 382)
            DiscriminatorBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2
            ),
            # shape (*, 128, 191, 191)
            DiscriminatorBlock(
                in_channels=128,
                out_channels=512,
                kernel_size=3,
                stride=1
            ),
            # shape (*, 512, 189, 189)
            DiscriminatorBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=2
            ),
            # shape (*, 512, 94, 94)
            DiscriminatorBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1
            ),
            # shape (*, 512, 92, 92)
            DiscriminatorBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=2
            ),
            # shape (*, 512, 45, 45)
            DiscriminatorBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1
            ),
            # shape (*, 512, 43, 43)
            DiscriminatorBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=2
            )
            # shape (*, 512, 21, 21)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(21*21*512, 1024),
            self.lrelu,
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x:torch.Tensor):
        x = self.in_conv(x)
        x = self.lrelu(x)
        x = self.blocks(x)
        x = self.fc(x)
        return x
    
class SRGAN(nn.Module):
    """The Super Resolution Generative Aversarial Network (SRGAN)

    Attributes
    ----------
    res_blocks : int
        The number of residual blocks of the Generator network

    channels : int
        The number of channels of the inputs images. Either 3 or 1.
    """
    def __init__(self, res_blocks:int=16, channels:int=3):
        super().__init__()
        self.res_blocks = res_blocks
        self.channels = channels
        self.generator = Generator(
            residual_blocks = res_blocks,
            channels=channels
        )
        
        self.discriminator = Discriminator(
            channels=channels
        )
    
    def discriminate(self, x:torch.Tensor):
        return self.discriminator(x)
    
    def generate(self, x:torch.Tensor):
        return self.generator(x)
    
    def loss(self, x_lr:torch.Tensor, x_hr:torch.Tensor):
        x_sr = self.generator(x_lr)
        content_loss = F.mse_loss(x_hr, x_sr).sum()
        adversarial_loss = -torch.log(self.discriminator(x_sr)).sum()
        perceptual_loss = content_loss + 1e-3 * adversarial_loss
        return perceptual_loss