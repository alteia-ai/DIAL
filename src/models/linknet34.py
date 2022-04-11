import torch.nn as nn
import torch


class SeparableConv2d_BN(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, bias=False):
        super(SeparableConv2d_BN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        self.BN1 = nn.BatchNorm2d(in_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.pointwise(x)
        x = self.BN2(x)
        return x


class Decoder(nn.Module):
    """LinkNet(34) decoder"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, drop=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(in_planes // 4),
                                   nn.ReLU(inplace=True), )
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(in_planes // 4, in_planes // 4, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes // 4, out_planes, 1, 1, 0, bias=bias),
                                   nn.BatchNorm2d(out_planes),
                                   nn.ReLU(inplace=True), )
        self.drop = drop

    def forward(self, x):
        if self.drop:
            x = nn.Dropout2d(p=0.05)(x)
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class BasicBlock(nn.Module):
    """Block used in ResNet18 and 34"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False, conv="classic"):
        super(BasicBlock, self).__init__()
        if conv == "classic":
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
            self.bn2 = nn.BatchNorm2d(out_planes)
        else:
            self.conv1 = SeparableConv2d_BN(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                                            bias=bias)
            self.conv2 = SeparableConv2d_BN(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
            self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride > 1 or in_planes == 12:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_planes), )
        self.conv_type = conv

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.conv_type == "classic":
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.conv_type == "classic":
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Encoder34(nn.Module):
    """Encoder for LinkNet34 (ie ResNet34)"""
    def __init__(self, layer, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False,
                 conv="classic", drop=False):
        super(Encoder34, self).__init__()
        self.layer = layer
        if self.layer == 1 or self.layer == 4:
            self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias, conv)
            self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block3 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
        if self.layer == 2:
            self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias, conv)
            self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block3 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block4 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
        if self.layer == 3:
            self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias, conv)
            self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block3 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block4 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block5 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block6 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
        self.drop =  drop 

    def forward(self, x):
        if self.drop:
            x = nn.Dropout2d(p=0.05)(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if self.layer == 2:
            x = self.block4(x)
        if self.layer == 3:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
        return x

class LinkNet34(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self, in_channels, n_classes, pretrain, drop=False, confidnet=False, disir = True):
        """
        Model initialization
        """
        super(LinkNet34, self).__init__()
        # assume one channel per class
        if disir:
            in_channels += n_classes
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder34(1, 64, 64, 3, 1, 1)
        self.encoder2 = Encoder34(2, 64, 128, 3, 2, 1)
        self.encoder3 = Encoder34(3, 128, 256, 3, 2, 1)
        self.encoder4 = Encoder34(4, 256, 512, 3, 2, 1)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)
        self.net_name = "LinkNet34_Revolver"
        self.apply(self.weight_init)
        self.drop = 0.1 if drop else 0
        self.confidnet = confidnet

    def forward(self, x):  # , fg_mean, bg_mean

        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks

        d4 = e3 + self.decoder4(e4)
        if self.drop:
            d4 = nn.functional.dropout2d(d4, self.drop)
        d3 = e2 + self.decoder3(d4)
        if self.drop:
            d3 = nn.functional.dropout2d(d3, self.drop)
        d2 = e1 + self.decoder2(d3)
        if self.drop:
            d2 = nn.functional.dropout2d(d2, self.drop)
        d1 = x + self.decoder1(d2)
        if self.drop:
            d1 = nn.functional.dropout2d(d1, self.drop)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        final = self.tp_conv2(y)

        # final = self.lsm(y)
        if self.confidnet:
            return final, y
        else:
            return final
