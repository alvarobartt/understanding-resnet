"""ResNet20 Traning with CIFAR10 (Modified, not original version)

He, Kaiming, et al. 'Deep Residual Learning for Image Recognition'
https://arxiv.org/pdf/1512.03385.pdf
"""

from resnet import ResNet

import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.datasets import CIFAR10


def train_resnet20_with_cifar10():
    # Check that GPU support is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiliaze ResNet20 for CIFAR10 and move it to the GPU (CPU if not available)
    model = ResNet(blocks=[3, 3, 3], filters=[16, 32, 64], num_classes=10)
    model.to(device)
    
    # Count the total number of trainable parameters
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"# of Trainable Parameters: {trainable_parameters}")

    # Define the mean/std normalization values (https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#gistcomment-2851662)
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std= (0.2470, 0.2435, 0.2616)

    # Initialize/Define train transformation
    train_transform = T.Compose([
        T.Pad(padding=4), # Can be removed and replaced with T.RandomCrop(size=(32, 32), padding=4)
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=(32, 32)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Initialize/Define test transformation
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Define the batch size before preparing the dataloaders
    BATCH_SIZE = 128

    # Load CIFAR10 train dataset (transform it too), and initialize dataloader
    train_dataset = CIFAR10(root="data", train=True, download=True, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load CIFAR10 test dataset (transform it too), and initialize dataloader
    test_dataset = CIFAR10(root="data", train=False, download=True, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define training parameters
    ITERATIONS = 64000
    EPOCHS = ITERATIONS//len(train_dataloader)
    LR_MILESTONES = [32000//len(train_dataloader), 48000//len(train_dataloader)]

    # Define the loss function, the optimizer, and the learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=0.1)

    # Weights and Biases Logging? -> TODO

    # Initialize variables before training
    smaller_test_error = 1.0

    # Training loop
    # List of TODOs
    # - Train/Val split (45k/5k)
    # - Include Kaiming He weight initilization
    for epoch in range(1, EPOCHS+1):
        # best_acc = .0
        # print(f"\nEpoch {epoch}/{EPOCHS}\n{'='*25}")
        # for phase in ['train', 'val']:
        #     running_loss = .0
        #     running_corrects = .0
        #     if phase == 'train': model.train()
        #     if phase == 'val': model.eval()
        #     for inputs, labels in loaders[phase]:
        #         inputs, labels = inputs.to(device), labels.to(device)

        #         optimizer.zero_grad()

        #         with torch.set_grad_enabled(phase == 'train'):
        #             outputs = model(inputs)
        #             _, preds = torch.max(outputs, 1)
        #             loss = criterion(outputs, labels)
                    
        #             if phase == 'train':
        #                 loss.backward()
        #                 optimizer.step()

        #         running_loss += loss.item() * inputs.size(0)
        #         running_corrects += torch.sum(preds == labels)
        #     epoch_loss = running_loss / dataset_sizes[phase]
        #     epoch_acc = running_corrects.double() / dataset_sizes[phase]
        #     if phase == 'train': scheduler.step()
        #     if phase == 'val' and epoch_acc > best_acc:
        #         best_acc = epoch_acc
        #         best_model_weights = deepcopy(model.state_dict())
        #     print(f"Loss ({phase}): {epoch_loss}, Acc ({phase}): {epoch_acc}")


if __name__ == '__main__':
    train_resnet20_with_cifar10()
