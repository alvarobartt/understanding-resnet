import torch.nn as nn
import torch.nn.functional as F

from resnet_block import ResNetBasicBlock


class ResNetNN(nn.Module):
    
    def __init__(self):
        super(ResNetNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        
        # Input image shape is 32x32 (no downsampling)
        self.rb1 = nn.Sequential(
            ResNetBasicBlock(in_channels=16, out_channels=16, stride=1),
            ResNetBasicBlock(in_channels=16, out_channels=16, stride=1),
            ResNetBasicBlock(in_channels=16, out_channels=16, stride=1)
        )

        # Input image shape is 32x32 (downsampling in the first layer of the block, 32x32 -> 16x16)
        self.rb2 = nn.Sequential(
            ResNetBasicBlock(in_channels=16, out_channels=32, stride=2), # "The subsampling is performed by convolutions with a stride of 2"
            ResNetBasicBlock(in_channels=32, out_channels=32, stride=1),
            ResNetBasicBlock(in_channels=32, out_channels=32, stride=1)
        )

        # Input image shape is 16x16 (downsampling in the first layer of the block, 16x16 -> 8x8)
        self.rb3 = nn.Sequential(
            ResNetBasicBlock(in_channels=32, out_channels=64, stride=2), # "The subsampling is performed by convolutions with a stride of 2"
            ResNetBasicBlock(in_channels=64, out_channels=64, stride=1),
            ResNetBasicBlock(in_channels=64, out_channels=64, stride=1)
        )
        
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[3])
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x
 

if __name__ == "__main__":
    net = ResNetNN()
    print(net)
    
    import torch
    
    x = torch.randn((1, 3, 32, 32))
    y = net(x)
    
    print(x.shape, y.shape)
    print(y)