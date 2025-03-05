import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class ResNeXtBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride, cardinality=8):
#         super(ResNeXtBlock, self).__init__()
#         bottleneck_channels = out_channels // 2  
#         self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(bottleneck_channels)
#         self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
#                                stride=stride, padding=1, groups=cardinality, bias=False)
#         self.bn2 = nn.BatchNorm2d(bottleneck_channels)
#         self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
    
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         return F.relu(out)

# class ResNeXt(nn.Module):
#     def __init__(self, block, block_counts, scale_factor=3.75, cardinality=8, num_classes=10):
#         super(ResNeXt, self).__init__()
#         self.in_channels = int(64 * scale_factor)  # 初始通道数：64 * 3.75 = 240
#         self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         # 三个 stage，输出通道依次为：240, 480, 960
#         self.layer1 = self._make_layer(block, int(64 * scale_factor), block_counts[0], stride=1, cardinality=cardinality)
#         self.layer2 = self._make_layer(block, int(128 * scale_factor), block_counts[1], stride=2, cardinality=cardinality)
#         self.layer3 = self._make_layer(block, int(256 * scale_factor), block_counts[2], stride=2, cardinality=cardinality)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(int(256 * scale_factor), num_classes)
    
#     def _make_layer(self, block, out_channels, num_blocks, stride, cardinality):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for s in strides:
#             layers.append(block(self.in_channels, out_channels, s, cardinality))
#             self.in_channels = out_channels
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out

# def ResNeXt_CIFAR(num_classes=10, scale_factor=3.75, cardinality=8, block_counts=[3,3,3]):
#     return ResNeXt(ResNeXtBlock, block_counts, scale_factor=scale_factor, cardinality=cardinality, num_classes=num_classes)
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=10, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = None

        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 64, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 224, layers[3], num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear((224) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
          x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNeXt_CIFAR(**kwargs):
    return ResNeXt(BasicBlock, [2, 2, 2, 2], **kwargs)
