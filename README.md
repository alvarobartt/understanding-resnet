# :bookmark_tabs: understanding-resnet

![image](https://user-images.githubusercontent.com/36760800/110832871-142dff80-829c-11eb-9c13-01d417e535d2.png)

## :crystal_ball: Future Tasks

- [ ] Plain CNN vs ResNet
- [ ] VGG vs ResNet in IM and OD

## Implementation

In order to understand how does the ResNet architecture work, we will be implementing the simplest version of it, which
is the ResNet20 for CIFAR10. This exercise will be useful to understand the main differences between a plain convolutional
neural network and residual network.

__Main considerations__:

- Input images are 32x32px (width x height) in RGB format (3 channels), which is a Tensor of shape `torch.Tensor([3, 32, 32])`.

- The dataset consists of 50k training images and 10k test images, classified in 10 classes.

- The data will be augmented padding 4px on each side, followed by a random crop of a window of shape 32x32 either from the 
original image or from its horizontal flip; just for the training data.

- The architecture is summarized in the following table, where `n=3` leading to a neural network with 20 weighted layers.

  | output map size | 32 x 32 | 16 x 16 | 8 x 8 |
  |-----------------|---------|---------|-------|
  | # layers        | 1 + 2n  | 2n      | 2n    |
  | # filters       | 16      | 32      | 64    |
  
- Both, the plain neural network and the residual neural network, have the exact same architecture.

- The numbers of the convolutional filters are 16, 32, and 64; so that the size of the input image goes 
from 32x32 to 16x16, and then to 8x8.

- The neural network starts off with a convolutional layer which applies a 3x3 convolution, resulting in 
16 out channels.

- Then we will include the residual blocks (the basic, not the bottleneck ones), that is a stack of `6*n`
layers with 3x3 convolutions, that contains `2n` layers for each feature map size.

- The subsampling is performed by convolutions with a stride of 2.

- Finally, the neural network ends with a global average pooling and a fully connected linear layer with 10 units
that stand for the 10 classes of the CIFAR10 dataset.

All this information can be found in the original paper in the section "_4.2. CIFAR-10 and Analysis_", that contains the 
experiments conducted by the authors on the CIFAR10 dataset.

## References

### From research papers

- [He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

### From books/posts/etc.

- [Howard, J., Gugger, S., &amp; Chintala, S. (2020). Chapter 14. ResNets. In Deep learning for coders with fastai and PyTorch: AI applications without a PhD (pp. 441–458). O'Reilly Media, Inc.](https://www.amazon.es/Deep-Learning-Coders-Fastai-Pytorch/dp/1492045527)

## Curated implementations

- [Official PyTorch implementation of ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
- [PyTorch Tutorial to implement a ResNet to train with CIFAR10](https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/)
- [Un-official implementation of ResNet for CIFAR10/100 by @akamaster](https://github.com/akamaster/pytorch_resnet_cifar10)
- [Un-official implementation of ResNet for CIFAR10 by @kuanglui](https://github.com/kuangliu/pytorch-cifar)
