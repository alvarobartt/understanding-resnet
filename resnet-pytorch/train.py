"""ResNet Traning with CIFAR10

He, Kaiming, et al. 'Deep Residual Learning for Image Recognition'
https://arxiv.org/pdf/1512.03385.pdf
"""

from __future__ import absolute_import

import os

import click

import json

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

from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from utils import select_device, count_trainable_parameters, count_layers
from utils import MEAN_NORMALIZATION, STD_NORMALIZATION

MODELS = {
    'resnet20': {
        'a': resnet20(zero_padding=False, pretrained=False),
        'b': resnet20(zero_padding=True, pretrained=False)
    },
    'resnet32': {
        'a': resnet32(zero_padding=False, pretrained=False),
        'b': resnet32(zero_padding=True, pretrained=False)
    },
    'resnet44': {
        'a': resnet44(zero_padding=False, pretrained=False),
        'b': resnet44(zero_padding=True, pretrained=False)
    },
    'resnet56': {
        'a': resnet56(zero_padding=False, pretrained=False),
        'b': resnet56(zero_padding=True, pretrained=False)
    },
    'resnet101': {
        'a': resnet110(zero_padding=False, pretrained=False),
        'b': resnet110(zero_padding=True, pretrained=False)
    }
}

@click.command()
@click.option('-a', '--arch', required=True, type=click.Choice(list(MODELS.keys()), case_sensitive=False))
@click.option('-z', '--zero-padding', is_flag=True, default=False)
def train_cifar10(arch: str, zero_padding: bool) -> None:
    """Trains any ResNet with CIFAR10."""
    # Initializes the selected model
    option = 'a' if zero_padding else 'b'
    model = MODELS[arch][option]

    # Check whether GPU support is available or not
    device = torch.device(select_device())

    # Initiliaze ResNet for CIFAR10 and move it to the GPU (CPU if not available)
    # model = nn.DataParallel(model) => https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
    model.to(device)

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # https://pytorch.org/docs/stable/backends.html
    if torch.backends.cudnn.is_available(): torch.backends.cudnn.benchmark = True
    
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
    # Loss functions usually don't need to be moved to CUDA: https://discuss.pytorch.org/t/move-the-loss-function-to-gpu/20060/3
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
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
    config.batches_train = len(train_dataloader)
    config.batches_test = len(test_dataloader)
    config.architecture = arch
    config.layers = total_layers
    config.parameters = trainable_parameters
    config.option = option
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

    # Define best prec@1 before training
    best_prec1 = 0.

    # Training loop with wandb logging
    wandb.watch(model)
    for epoch in range(1, EPOCHS+1):
        # Initialize AverageMeters before training
        train_losses = AverageMeter()
        train_prec1 = AverageMeter()

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

            prec1 = accuracy(outputs.data, labels)[0]
            train_losses.update(loss.item(), inputs.size(0))
            train_prec1.update(prec1.item(), inputs.size(0))
        
        train_loss = train_losses.avg
        train_acc = train_prec1.avg / 100
        train_error = 1.0 - train_acc
        train_time = time() - start_time

        wandb.log({
            'train_loss': train_loss, 'train_acc': train_acc,
            'train_error': train_error, 'train_time': train_time
        }, step=epoch)

        scheduler.step()

        # Initialize AverageMeters before evaluation
        test_losses = AverageMeter()
        test_prec1 = AverageMeter()

        model.eval()
        start_time = time()

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                outputs, loss = outputs.float(), loss.float()

                prec1 = accuracy(outputs.data, labels)[0]
                test_losses.update(loss.item(), inputs.size(0))
                test_prec1.update(prec1.item(), inputs.size(0))
            
            test_loss = test_losses.avg
            test_acc = test_prec1.avg / 100
            test_error = 1.0 - test_acc
            test_time = time() - start_time
            
            wandb.log({
                'test_loss': test_loss, 'test_acc': test_acc,
                'test_error': test_error, 'test_time': test_time
            }, step=epoch)

        if best_prec1 is None: best_prec1 = test_acc
        if best_prec1 <= test_acc:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"{arch}{option}-cifar10.pth"))
            with open(os.path.join(wandb.run.dir, f"{arch}{option}-cifar10.json"), "w") as f:
                json.dump({"epoch": epoch, "train_prec1": train_acc, "test_prec1": test_acc}, f)
            best_prec1 = test_acc

    # Finish logging this wandb run
    # https://docs.wandb.ai/library/init#how-do-i-launch-multiple-runs-from-one-script
    run.finish()


if __name__ == '__main__':
    train_cifar10()
