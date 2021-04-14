"""ResNet20 Traning with CIFAR10 (Modified, not original version)

He, Kaiming, et al. 'Deep Residual Learning for Image Recognition'
https://arxiv.org/pdf/1512.03385.pdf
"""

import os

from time import time

from math import ceil

import wandb

import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from resnet import ResNet, resnet20, resnet32


def train_resnet_cifar10(model: ResNet, model_name: str) -> None:
    # Check that GPU support is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiliaze ResNet for CIFAR10 and move it to the GPU (CPU if not available)
    model.to(device)
    
    # Count the total number of trainable parameters
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"# of Trainable Parameters: {trainable_parameters}")

    # Define the mean/std normalization values (https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#gistcomment-2851662)
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std= (0.247, 0.2435, 0.2616)

    # Initialize/Define train transformation
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=(32, 32), padding=4),
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
    EPOCHS = ceil(ITERATIONS/len(train_dataloader))
    LR_MILESTONES = [
        ceil(32000/len(train_dataloader)),
        ceil(48000/len(train_dataloader))
    ]

    # Define the loss function, the optimizer, and the learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=9e-1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=1e-1)

    # Start a new wandb run
    # https://docs.wandb.ai/library/init#how-do-i-launch-multiple-runs-from-one-script
    run = wandb.init(project='resnet-pytorch', entity='alvarobartt', reinit=True)
    
    # Save the configuration for the current wandb run
    config = wandb.config
    config.iterations = ITERATIONS
    config.epochs = EPOCHS
    config.batch_size = BATCH_SIZE
    config.architecture = model_name
    config.dataset = 'cifar10'
    config.transformations = True
    config.input_shape = '[32,32,3]'
    config.criterion = 'cross_entropy_loss'
    config.optimizer = 'sgd'
    config.learning_rate = 1e-1
    config.channels_last = False
    config.changelog = 'removed softmax in last fc layer and used AdaptiveAvgPool2d'

    # Initialize variables before training
    best_error = None

    # Training loop with wandb logging
    wandb.watch(model)
    for epoch in range(1, EPOCHS+1):
        running_loss, running_corrects = .0, .0
        
        model.train()
        start_time = time()

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += (loss.item() * inputs.size(0))
            running_corrects += torch.sum(preds == labels)
        
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)
        train_error = 1.0 - train_acc
        train_time = time() - start_time

        wandb.log({
            'train_loss': train_loss, 'train_acc': train_acc,
            'train_error': train_error, 'train_time': train_time
        }, step=epoch)

        scheduler.step()

        model.eval()
        start_time = time()

        with torch.no_grad():
            running_loss, running_corrects = .0, .0

            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += (loss.item() * inputs.size(0))
                running_corrects += torch.sum(preds == labels)
            
            test_loss = running_loss / len(test_dataset)
            test_acc = running_corrects.double() / len(test_dataset)
            test_error = 1.0 - test_acc
            test_time = time() - start_time
            
            wandb.log({
                'test_loss': test_loss, 'test_acc': test_acc,
                'test_error': test_error, 'test_time': test_time
            }, step=epoch)

        if best_error is None: best_error = test_error
        if best_error >= test_error:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"{model_name}-cifar10.pth"))
            best_error = test_error

    # Finish logging this wandb run
    # https://docs.wandb.ai/library/init#how-do-i-launch-multiple-runs-from-one-script
    run.finish()


if __name__ == '__main__':
    train_resnet_cifar10(model=resnet20(), model_name='resnet20')
    train_resnet_cifar10(model=resnet32(), model_name='resnet32')
