# :alembic: Implementation

In order to understand how does the ResNet architecture works, we will be implementing it 
in PyTorch according to the original paper. So on, here you'll find the implementation of
both ResNets for CIFAR10 and for ImageNet.

Before going through the implementation, reading both the [original paper](https://arxiv.org/abs/1512.03385) 
and the notes in [README.md](https://github.com/alvarobartt/understanding-resnet/blob/master/README.md) is 
recommended, so as to have a better understanding on how does this architecture work.

---

## :test_tube: ResNets for CIFAR10

### :open_file_folder: Dataset

* Input images are 32x32px (width x height) in RGB format (3 channels), which is a Tensor of shape 
`torch.Tensor([3, 32, 32])` (assuming `channels_last=False`).

* The dataset consists of 50k training images and 10k test images, classified in 10 classes.

* The data will be augmented padding 4px on each side, followed by a random crop of a window of shape 32x32 
either from the original image or from its horizontal flip; just for the training data.

:pushpin: A team led by researchers at MIT's Computer Science and Artificial Intelligence Lab (CSAIL) looked at 10 major 
datasets that have been cited over 100,000 times and that include CIFAR and ImageNet. They found a 3.4% average error 
rate across all datasets, including 6% for ImageNet, which is arguably the most widely used dataset for popular image 
recognition systems. Check the CIFAR10 label errors at [Label Errors: Explore CIFAR10](https://labelerrors.com/).

### :brain: Architecture

* The architecture is summarized in the following table, where `n=3` (ResNet20), `n=5` (ResNet32), `n=7` (ResNet44), `n=9` (ResNet56), and `n=18` (ResNet110).

  | output map size | 32 x 32 | 16 x 16 | 8 x 8 |
  |-----------------|---------|---------|-------|
  | # layers        | 1 + 2n  | 2n      | 2n    |
  | # filters       | 16      | 32      | 64    |
  
* Both, the plain neural network and the residual neural network, have the exact same architecture, besides
the identity shortcut in the second one; but the same amount of trainable parameters and layers.

* The convolutional filters to be applied are 16, 32, and 64; so that the size of the input image goes 
from 32x32 to 16x16, and then to 8x8.

* The neural network starts off with a convolutional layer which applies a 3x3 convolution, resulting in 
16 out channels.

* Then we will include the residual blocks (the basic, not the bottleneck ones), that is a stack of `6*n`
layers with 3x3 convolutions, that contains `2n` layers for each feature map size.

* The subsampling/downsampling is performed by convolutions with a stride of 2, instead of using pooling operations. 
A comparison between both approaches can be found at: 
[Stackexchange: Pooling vs. stride for downsampling](https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling/387522)

* There are 2 known options for the downsampling/subsampling in the basic blocks as described in the paper:
  * Option A: the shortcut performs identity mapping with extra zero entries padded, to increase the dimensions.
  * Option B: the projection shortcut applies 1x1 convolutions to match the dimensions.

* The main difference between both options is that the option A does not include extra parameters, which
is nice in order to compare the residual networks versus their plain counterpart. And, the option B includes a
few convolutional layers, which means more parameters than its plain counterpart.

* Finally, the neural network ends with a global average pooling and a fully connected linear layer with 10 units
that stands for the 10 classes of the CIFAR10 dataset.

### :mechanical_arm: Training

```bash
>>> python resnet-pytorch/train.py --help
Usage: train.py [OPTIONS]

  Trains any ResNet with CIFAR10.

Options:
  -a, --arch [resnet20|resnet32|resnet44|resnet56|resnet101]
                                  [required]
  -z, --zero-padding
  --help                          Show this message and exit.
```

e.g. to train ResNet20 with zero_padding, no extra parameter (option A):

```bash
>>> python resnet-pytorch/train.py --arch resnet20 --zero-padding
```

---

## :test_tube: ResNets for ImageNet

### :open_file_folder: Dataset

* The dataset consists of 1.28 million training images, 50k validation images, and 100 test images, classified in 1000 different classes.

* Both precision @ 1, and precision @ 5, are being calculated while testing the model.

* All the images are resized to their shorter side randomly sampled for scale augmentation, then a random crop of size 224x224px 
is sampled from either the original image or from its horizontal flip (just for the training images); and, finally, all the images 
are normalized in the range [-1, 1] with the mean and std.

* So that the input images end up being 224x224px (width x height) in RGB format (3 channels), which is a Tensor of shape 
`torch.Tensor([3, 32, 32])` (assuming `channels_last=False`).

:pushpin: A team led by researchers at MIT's Computer Science and Artificial Intelligence Lab (CSAIL) looked at 10 major 
datasets that have been cited over 100,000 times and that include CIFAR and ImageNet. They found a 3.4% average error 
rate across all datasets, including 6% for ImageNet, which is arguably the most widely used dataset for popular image 
recognition systems. Check the ImageNet label errors at [Label Errors: Explore ImageNet](https://labelerrors.com/).

### :brain: Architecture

* The architecture is summarized in the following table:

<img width="913" alt="imagen" src="https://user-images.githubusercontent.com/36760800/117533350-9c0a5e80-afec-11eb-8992-6154fe4cead8.png">
  
* The neural network starts off with a convolutional layer which applies a 7x7 convolution with a stride of 2 so as to
reduce the dimension of the input image, resulting in 64 out channels.

* Also before applying the stack of residual blocks, it performs a 3x3 max pooling also with a stride of 2, to further reduce
the dimensions of the image.

* Both ResNet-18, and ResNet-34 consist on a stack of `BasicBlocks` to perform the identity shortcuts, while
ResNet-50, ResNet-101, and ResNet-152 consist on a stack of `BottleneckBlocks`, since those are more optimal
and less computatinoally expensive as they are using 1x1 convolutions to perform the upsample/downsample before
and after the 3x3 convolution, respectively.

* Finally, the neural network ends with a global average pooling and a fully connected linear layer with 1000 units
that stands for the 1000 classes of the ImageNet-2012 dataset.
