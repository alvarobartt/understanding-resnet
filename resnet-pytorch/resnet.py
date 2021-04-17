"""ResNet Implementation in PyTorch

He, Kaiming, et al. 'Deep Residual Learning for Image Recognition'
https://arxiv.org/pdf/1512.03385.pdf
"""

from typing import List, Tuple, Type, Union

import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch import Tensor

from torch.hub import load_state_dict_from_url


__all__ = ['BasicBlock', 'BottleneckBlock', 'ResNet']


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.subsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            # "The subsampling is performed by convolutions with a stride of 2"
            self.subsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels * self.expansion)
            )

    def forward(self, x: Tensor) -> Tensor:
        x_ = F.relu(self.bn1(self.conv1(x)))
        x_ = self.bn2(self.conv2(x_))
        x_ += self.subsample(x)
        x_ = F.relu(x_)
        return x_


class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * self.expansion)

        self.subsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.subsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels * self.expansion)
            )

    def forward(self, x: Tensor) -> Tensor:
        x_ = F.relu(self.bn1(self.conv1(x)))
        x_ = F.relu(self.bn2(self.conv2(x_)))
        x_ = self.bn3(self.conv3(x_))
        x_ += self.subsample(x)
        x_ = F.relu(x_)
        return x_


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, BottleneckBlock]], blocks: List[int], filters: List[int], num_classes: int) -> None:
        super(ResNet, self).__init__()

        assert len(blocks) == 3 or len(blocks) == 4, "ResNet can either have `3` blocks for CIFAR10, or `4` for ImageNet"
        assert len(blocks) == len(filters), "# of blocks must match # of filters"

        self.in_channels = filters[0]
        
        kernel_size = 3 if len(blocks) == 3 else 7
        stride = 1 if len(blocks) == 3 else 2
        padding = 1 if len(blocks) == 3 else 3

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=filters[0])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # RL stands for Residual Layer
        self.rl1 = self._make_layer(block=block, num_blocks=blocks[0], planes=filters[0], stride=1)
        self.rl2 = self._make_layer(block=block, num_blocks=blocks[1], planes=filters[1], stride=2)
        self.rl3 = self._make_layer(block=block, num_blocks=blocks[2], planes=filters[2], stride=2)
        if len(blocks) == 4: self.rl4 = self._make_layer(block=block, num_blocks=blocks[3], planes=filters[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(filters[-1] * block.expansion, num_classes)

        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.rl1(x)
        x = self.rl2(x)
        x = self.rl3(x)
        if hasattr(self, 'rl4'): x = self.rl4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) # Removed softmax: https://discuss.pytorch.org/t/resnet-last-layer-modification/33530/2
        return x

    def _make_layer(self, block: Type[Union[BasicBlock, BottleneckBlock]], num_blocks: int, planes: int, stride: int) -> nn.Sequential:
        layers = list()

        if stride != 1 or self.in_channels != planes * block.expansion:
            layers.append(block(in_channels=self.in_channels, out_channels=planes, stride=stride))
        else:
            layers.append(block(in_channels=self.in_channels, out_channels=planes, stride=1))

        self.in_channels = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(in_channels=self.in_channels, out_channels=planes, stride=1))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d): init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
 

def resnet20(pretrained=False) -> ResNet:
    model = ResNet(block=BasicBlock, blocks=[3, 3, 3], filters=[16, 32, 64], num_classes=10)
    if pretrained: model.load_state_dict(load_state_dict_from_url("https://github.com/alvarobartt/understanding-resnet/releases/download/v0.1-cifar10/resnet20-cifar10.pth"))
    return model

def resnet32(pretrained=False) -> ResNet:
    model = ResNet(block=BasicBlock, blocks=[5, 5, 5], filters=[16, 32, 64], num_classes=10)
    if pretrained: model.load_state_dict(load_state_dict_from_url("https://github.com/alvarobartt/understanding-resnet/releases/download/v0.1-cifar10/resnet32-cifar10.pth"))
    return model

def resnet44(pretrained=False) -> ResNet:
    model = ResNet(block=BasicBlock, blocks=[7, 7, 7], filters=[16, 32, 64], num_classes=10)
    if pretrained: raise NotImplementedError
    return model

def resnet56(pretrained=False) -> ResNet:
    model = ResNet(block=BasicBlock, blocks=[9, 9, 9], filters=[16, 32, 64], num_classes=10)
    if pretrained: raise NotImplementedError
    return model

def resnet18(pretrained=False) -> ResNet:
    model = ResNet(block=BasicBlock, blocks=[2, 2, 2, 2], filters=[64, 128, 256, 512], num_classes=1000)
    if pretrained: model.load_state_dict(load_state_dict_from_url("https://github.com/alvarobartt/understanding-resnet/releases/download/v0.1-imagenet/resnet18-imagenet-ported.pth"))
    return model

def resnet34(pretrained=False) -> ResNet:
    model = ResNet(block=BasicBlock, blocks=[3, 4, 6, 3], filters=[64, 128, 256, 512], num_classes=1000)
    if pretrained: raise NotImplementedError
    return model

def resnet50(pretrained=False) -> ResNet:
    model = ResNet(block=BottleneckBlock, blocks=[3, 4, 6, 3], filters=[64, 128, 256, 512], num_classes=1000)
    if pretrained: raise NotImplementedError
    return model


if __name__ == "__main__":
    # CIFAR10 ResNet20
    model = resnet20()
    print(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    x = torch.randn((1, 3, 32, 32))
    y = model(x)
    
    print(x.shape, y.shape)
    print(sum(param.numel() for param in model.parameters() if param.requires_grad))

    # ImageNet ResNet18
    model = resnet50()
    print(model)
    
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
    
    print(x.shape, y.shape)
    print(sum(param.numel() for param in model.parameters() if param.requires_grad))
