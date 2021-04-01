"""ResNet

He, Kaiming, et al. 'Deep Residual Learning for Image Recognition'
https://arxiv.org/pdf/1512.03385.pdf
"""

from typing import List

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        x_ = F.relu(self.bn1(self.conv1(x)))
        x_ = self.bn2(self.conv2(x_))
        x_ += self.shortcut(x)
        x_ = F.relu(x_)
        return x_


class ResNet(nn.Module):
    def __init__(self, blocks: List[int], filters: List[int], num_classes: int) -> None:
        super(ResNet, self).__init__()

        assert len(blocks) == 3, "ResNet for CIFAR10 uses just 3 residual layers (ImageNet uses 4)"
        assert len(blocks) == len(filters), "# of blocks must match # of filters"

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=filters[0])
        
        # Input image shape is 32x32 (no subsampling/downsampling)
        self.rl1 = self._make_layers(num_blocks=blocks[0], planes=filters[0], subsampling=False)

        # Input image shape is 32x32 (subsampling/downsampling in the first layer of the block, 32x32 -> 16x16)
        self.rl2 = self._make_layers(num_blocks=blocks[1], planes=filters[0], subsampling=True)

        # Input image shape is 16x16 (subsampling/downsampling in the first layer of the block, 16x16 -> 8x8)
        self.rl3 = self._make_layers(num_blocks=blocks[2], planes=filters[1], subsampling=True)
        
        self.fc1 = nn.Linear(filters[2], num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.rl1(x)
        x = self.rl2(x)
        x = self.rl3(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[3])
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x

    def _make_layers(self, num_blocks: int, planes: int, subsampling: bool) -> nn.Sequential:
        layers = list()
        
        # "The subsampling is performed by convolutions with a stride of 2"
        if subsampling:
            layers.append(BasicBlock(in_channels=planes, out_channels=planes*2, stride=2))
            num_blocks -= 1
            planes *= 2

        for _ in range(num_blocks):
            layers.append(BasicBlock(in_channels=planes, out_channels=planes, stride=1))

        return nn.Sequential(*layers)
 

if __name__ == "__main__":
    net = ResNet(blocks=[3, 3, 3], filters=[16, 32, 64], num_classes=10)
    print(net)
    
    import torch
    
    x = torch.randn((1, 3, 32, 32))
    y = net(x)
    
    print(x.shape, y.shape)
    print(sum(param.numel() for param in net.parameters() if param.requires_grad))
