""" Parts of the U-Net model """

from numpy.lib.function_base import diff
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)





class UNet(nn.Module):
    def __init__(self, n_classes, bilinear=True, nb_block = 4):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.nb_block = nb_block

        self.down_first = []
        for k in range(nb_block-1):
          self.down_first.append(Down(2**(6+k), 2**(6+k+1)))
        self.down_first = nn.ModuleList(self.down_first)

        factor = 2 if bilinear else 1
        self.down_last = Down(2**(6+nb_block-1), 2**(6+nb_block)//factor)

        self.up = []
        for k in range(nb_block-1):
          self.up.append(Up(2**(6+nb_block-k), 2**(6+nb_block-k-1)//factor, bilinear))

        self.up.append(Up(2**7, 2**6, bilinear))
        self.up = nn.ModuleList(self.up)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x1 = self.inc(x)

        list_x = [x]
        for k in range(self.nb_block-1):
          list_x.append(self.down_first[k](list_x[-1]))
        list_x.append(self.down_last(list_x[-1]))
        x = self.up[0](list_x[-1],list_x[-2])
        for k in range(1,self.nb_block):
          x = self.up[k](x, list_x[self.nb_block-k-1])
          
        logits = self.outc(x)
        return logits


####### SAME THING BUT 1D ##############

class DoubleConv1D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down1D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv1D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv1D(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2) )

        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, n_classes, bilinear=False, nb_block = 4):
        super().__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.nb_block = nb_block

        self.down_first = []
        for k in range(nb_block-1):
          self.down_first.append(Down1D(2**(6+k), 2**(6+k+1)))
        self.down_first = nn.ModuleList(self.down_first)

        factor = 2 if bilinear else 1
        self.down_last = Down1D(2**(6+nb_block-1), 2**(6+nb_block)//factor)

        self.up = []
        for k in range(nb_block-1):
          self.up.append(Up1D(2**(6+nb_block-k), 2**(6+nb_block-k-1)//factor, bilinear))

        self.up.append(Up1D(2**7, 2**6, bilinear))
        self.up = nn.ModuleList(self.up)
        self.outc = OutConv1D(64, n_classes)

    def forward(self, x):
        list_x = [x]
        for k in range(self.nb_block-1):
          list_x.append(self.down_first[k](list_x[-1]))
        list_x.append(self.down_last(list_x[-1]))

        x = self.up[0](list_x[-1],list_x[-2])
        for k in range(1,self.nb_block):
          x = self.up[k](x, list_x[self.nb_block-k-1])
          
        logits = self.outc(x)
        return logits
