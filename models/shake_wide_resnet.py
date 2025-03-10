import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Shake-shake regularization
class ShakeShake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = x1.new(x1.size(0)).uniform_()
            alpha = alpha.view(-1, 1, 1, 1).expand_as(x1)
        else:
            alpha = x1.new_full((x1.size(0), 1, 1, 1), 0.5)
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = grad_output.new(grad_output.size(0)).uniform_()
        beta = beta.view(-1, 1, 1, 1).expand_as(grad_output)
        return beta * grad_output, (1 - beta) * grad_output, None

class ShakeBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ShakeBlock, self).__init__()
        self.equal_io = (inplanes == planes)
        if not self.equal_io:
            self.downsample = downsample if downsample is not None else nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

        self.branch1 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes)
        )
        self.branch2 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        # 使用 shake‑shake 组合两个分支
        out = ShakeShake.apply(h1, h2, self.training)
        residual = self.downsample(x) if self.downsample is not None else x
        out = out + residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, channels, layers, num_classes=10):
        self.inplanes = channels[0]
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(channels[2] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def Shake_Wide_ResNet_18(**kwargs):
    channels = [64, 96, 160]
    layers = [4, 4, 3]
    return ResNet(ShakeBlock, channels, layers, **kwargs)
