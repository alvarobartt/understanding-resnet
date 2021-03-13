import torch.nn as nn


class ResNetNN(nn.Module):
    """
    """

    def __init__(self):
        super(ResNetNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        # resnet block x 3 (16x16)
        # resnet block x 3 (32x32)
        # resnet block x 3 (64x64)
        self.fc1 = nn.Linear(64*8*8, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # resnet layers
        x = F.avg_pool2d(x, kernel_size=4)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x