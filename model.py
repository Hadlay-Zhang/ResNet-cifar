import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality=8):
        super(ResNeXtBlock, self).__init__()
        bottleneck_channels = out_channels // 2  
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNeXt(nn.Module):
    def __init__(self, block, block_counts, scale_factor=3.75, cardinality=8, num_classes=10):
        super(ResNeXt, self).__init__()
        self.in_channels = int(64 * scale_factor)  # 初始通道数：64 * 3.75 = 240
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        # 三个 stage，输出通道依次为：240, 480, 960
        self.layer1 = self._make_layer(block, int(64 * scale_factor), block_counts[0], stride=1, cardinality=cardinality)
        self.layer2 = self._make_layer(block, int(128 * scale_factor), block_counts[1], stride=2, cardinality=cardinality)
        self.layer3 = self._make_layer(block, int(256 * scale_factor), block_counts[2], stride=2, cardinality=cardinality)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(256 * scale_factor), num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride, cardinality):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, cardinality))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNeXt_CIFAR(num_classes=10, scale_factor=3.75, cardinality=8, block_counts=[3,3,3]):
    return ResNeXt(ResNeXtBlock, block_counts, scale_factor=scale_factor, cardinality=cardinality, num_classes=num_classes)
