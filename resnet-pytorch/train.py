"""ResNet20 Traning with CIFAR10 (Modified, not original version)

He, Kaiming, et al. 'Deep Residual Learning for Image Recognition'
https://arxiv.org/pdf/1512.03385.pdf
"""

from __future__ import absolute_import

import os

from time import time

from math import ceil

import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from timm.utils.metrics import AverageMeter, accuracy

from resnet import ResNet, resnet20
from utils import select_device, count_trainable_parameters, count_layers
from utils import MEAN_NORMALIZATION, STD_NORMALIZATION


def train_resnet_cifar10(model: ResNet, model_name: str) -> None:
    # Check that GPU support is available
    device = torch.device(select_device())

    # Initiliaze ResNet for CIFAR10 and move it to the GPU (CPU if not available)
    model.to(device)
    
    # Count the total number of trainable parameters
    trainable_parameters = count_trainable_parameters(model=model)
    print(f"# of Trainable Parameters: {trainable_parameters}")

    # Count the total number of layers
    total_layers = count_layers(model=model)
    print(f"# of Layers: {total_layers}")

    # Initialize/Define train transformation
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=(32, 32), padding=4),
        T.ToTensor(),
        T.Normalize(mean=MEAN_NORMALIZATION, std=STD_NORMALIZATION)
    ])

    # Initialize/Define test transformation
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN_NORMALIZATION, std=STD_NORMALIZATION)
    ])

    # Define the batch size before preparing the dataloaders
    BATCH_SIZE = 128

    # Load CIFAR10 train dataset (transform it too), and initialize dataloader
    train_dataset = CIFAR10(root="data", train=True, download=True, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # Load CIFAR10 test dataset (transform it too), and initialize dataloader
    test_dataset = CIFAR10(root="data", train=False, download=True, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Define training parameters as described in the original paper
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
    config.dataset_train_size = len(train_dataset)
    config.dataset_test_size = len(test_dataset)
    config.transformations = True
    config.input_shape = '[32,32]'
    config.channels_last = False
    config.criterion = 'cross_entropy_loss'
    config.optimizer = 'sgd'
    config.learning_rate = 1e-1
    config.learning_rate_milestones = LR_MILESTONES

    # Initialize AverageMeters before training
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    test_losses = AverageMeter()
    test_top1 = AverageMeter()

    # Define best TOP-1 accuracy
    best_top1 = 0.

    # Training loop with wandb logging
    wandb.watch(model)
    for epoch in range(1, EPOCHS+1):
        model.train()
        start_time = time()

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs, loss = outputs.float(), loss.float()

            top1 = accuracy(outputs.data, labels)[0]
            train_losses.update(loss.item(), inputs.size(0))
            train_top1.update(top1.item(), inputs.size(0))
        
        train_loss = train_losses.avg
        train_acc = train_top1.avg / 100
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
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                outputs, loss = outputs.float(), loss.float()

                top1 = accuracy(outputs.data, labels)[0]
                test_losses.update(loss.item(), inputs.size(0))
                test_top1.update(top1.item(), inputs.size(0))
            
            test_loss = test_losses.avg
            test_acc = test_top1.avg / 100
            test_error = 1.0 - test_acc
            test_time = time() - start_time
            
            wandb.log({
                'test_loss': test_loss, 'test_acc': test_acc,
                'test_error': test_error, 'test_time': test_time
            }, step=epoch)

        if best_top1 is None: best_top1 = test_acc
        if best_top1 <= test_acc:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"{model_name}-cifar10.pth"))
            best_top1 = test_acc

    # Finish logging this wandb run
    # https://docs.wandb.ai/library/init#how-do-i-launch-multiple-runs-from-one-script
    run.finish()


if __name__ == '__main__':
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # https://pytorch.org/docs/stable/backends.html
    if torch.backends.cudnn.is_available(): torch.backends.cudnn.benchmark = True

    train_resnet_cifar10(model=resnet20(), model_name='resnet20')
