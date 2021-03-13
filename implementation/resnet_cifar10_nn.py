import torch.nn as nn
import torch.nn.functional as F

from resnet_block import ResNetBlock


class ResNetNN(nn.Module):
    """
    """

    def __init__(self):
        super(ResNetNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        
        self.rb1 = nn.Sequential(
            ResNetBlock(in_channels=16, out_channels=16, stride=1),
            ResNetBlock(in_channels=16, out_channels=16, stride=1),
            ResNetBlock(in_channels=16, out_channels=16, stride=1)
        )

        self.rb2 = nn.Sequential(
            ResNetBlock(in_channels=16, out_channels=32, stride=2),
            ResNetBlock(in_channels=32, out_channels=32, stride=1),
            ResNetBlock(in_channels=32, out_channels=32, stride=1)
        )

        self.rb3 = nn.Sequential(
            ResNetBlock(in_channels=32, out_channels=64, stride=2),
            ResNetBlock(in_channels=64, out_channels=64, stride=1),
            ResNetBlock(in_channels=64, out_channels=64, stride=1)
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


net = ResNetNN()
print(net)