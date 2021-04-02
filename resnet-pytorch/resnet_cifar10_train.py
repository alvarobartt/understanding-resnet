import torch
import torch.nn as nn
import torch.optim as optim

from resnet_cifar10_nn import ResNetNN

net = ResNetNN()

# Loss, optimizer and LR scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1) # Is this also iterations or epochs???

# Training conditions
TOTAL_EPOCHS = 64000 # This is iterations not epochs!!!!!!!! every epoch has N iter (= total_data / batch_size)
BATCH_SIZE = 128
VAL_SPLIT = 0.1
NUM_GPUS = 2 # I just have one GPU ¯\_( ❛ ͜ʖ ❛ )_/¯

#########################################
# LOG TRAINING USING WEIGHTS AND BIASES #
#########################################