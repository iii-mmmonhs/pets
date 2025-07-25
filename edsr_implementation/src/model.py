import torch
import torch.nn as nn
import torch.nn.init as init

class ResidualBlock(nn.Module):
    """
    Conv -> ReLU -> Conv, skip connection.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        return out


class EDSR(nn.Module):
    """
    Реализация модели.
    """
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_blocks=16):
        super(EDSR, self).__init__()
        self.scale_factor = scale_factor
        self.num_features = num_features

        self.conv_input = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        self.conv_mid = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Апскейл
        if scale_factor == 2:
             # Один шаг PixelShuffle для 2x
            self.upscale_conv = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.upscale_relu = nn.ReLU()
        elif scale_factor == 4:
            # Два шага PixelShuffle для 4x
            self.upscale_conv1 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.upscale_relu1 = nn.ReLU()
            self.upscale_conv2 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
            self.pixel_shuffle2 = nn.PixelShuffle(2)
            self.upscale_relu2 = nn.ReLU()
        else:
             # Других нет в датасете
             raise NotImplementedError("Поддерживаются только scale_factor 2 или 4")

        # Возвращаем к 3 каналам (RGB)
        self.conv_output = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Инициализация по Kaiming для ReLU.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out

        out = self.residual_blocks(out)
        out = self.conv_mid(out)
        out = out + residual

        # Апскейл
        if self.scale_factor == 2:
            out = self.upscale_conv(out)
            out = self.pixel_shuffle(out)
            out = self.upscale_relu(out)
        elif self.scale_factor == 4:
            out = self.upscale_conv1(out)
            out = self.pixel_shuffle1(out)
            out = self.upscale_relu1(out)
            out = self.upscale_conv2(out)
            out = self.pixel_shuffle2(out)
            out = self.upscale_relu2(out)

        out = self.conv_output(out)

        return out