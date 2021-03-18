# understanding-resnet

TL;DR In Residual Learning the layers are reformulated as learning residual functions with
reference to the layer inputs. These networks are easier to optimize, and can gain accuracy
from considerably increased depth. Along this repository not just an explanation is provided
but also a minimal implementation of a small ResNet for CIFAR10 with 20 layers.

## :crystal_ball: Future Tasks

* [ ] Plain CNN vs ResNet

* [ ] Take inspiration from https://learnopencv.com/understanding-alexnet/

* [ ] VGG vs ResNet in Image Classification and Object Detection

## :notebook: Explanation

Answering to the question "_Is learning better networks as easy as stacking more layers?_", so it's 
not as straight forward since we run into the problem of vanishing/exploding gradients, which is usually
addressed with batch normalization, so that those networks start converging for SGD with backprop.

Anyway, when those networks start converging, the degradation problem araises; so that the accuracy
gets saturated and then degrades rapidly, due to an increase in the training error (not overfitting).

The authors address the degradation problem introducing the concept of residual learning, which introduces
the idea of the residual mapping, where that there's no expectation that the stacked layers will fit
the underlying mapping, but the residual one. So that they state that it should be easier to optimize
the residual mapping than the unreferenced one.

So on, the "shortcut connections" are the ones connecting the input of a stack of convolutional layers
and the last convolutional layer on the stack, via skipping the intermediate connections.

![image](https://user-images.githubusercontent.com/36760800/110832871-142dff80-829c-11eb-9c13-01d417e535d2.png)

## :test_tube: Implementation

In order to understand how does the ResNet architecture work, we will be implementing the simplest version of it, which
is the ResNet20 for CIFAR10. This exercise will be useful to understand the main differences between a plain convolutional
neural network and residual network.

So on, before proceeding with the implementation, we will carefully read the research paper so as to extract some knowledge
required so as to properly understand how did the authors implement it.

### :open_file_folder: Dataset

* Input images are 32x32px (width x height) in RGB format (3 channels), which is a Tensor of shape `torch.Tensor([3, 32, 32])`.

* The dataset consists of 50k training images and 10k test images, classified in 10 classes.

* The data will be augmented padding 4px on each side, followed by a random crop of a window of shape 32x32 either from the 
original image or from its horizontal flip; just for the training data.

### :brain: Architecture

* The architecture is summarized in the following table, where `n=3` leading to a neural network with 20 weighted layers.

  | output map size | 32 x 32 | 16 x 16 | 8 x 8 |
  |-----------------|---------|---------|-------|
  | # layers        | 1 + 2n  | 2n      | 2n    |
  | # filters       | 16      | 32      | 64    |
  
* Both, the plain neural network and the residual neural network, have the exact same architecture.

* The numbers of the convolutional filters are 16, 32, and 64; so that the size of the input image goes 
from 32x32 to 16x16, and then to 8x8.

* The neural network starts off with a convolutional layer which applies a 3x3 convolution, resulting in 
16 out channels.

* Then we will include the residual blocks (the basic, not the bottleneck ones), that is a stack of `6*n`
layers with 3x3 convolutions, that contains `2n` layers for each feature map size.

* The subsampling/downsampling is performed by convolutions with a stride of 2, instead of using pooling operations. 
A comparison between both approaches can be found at: 
[Stackexchange: Pooling vs. stride for downsampling](https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling/387522)

* Finally, the neural network ends with a global average pooling and a fully connected linear layer with 10 units
that stand for the 10 classes of the CIFAR10 dataset.

  ---

All this information can be found in the original paper in the section "_4.2. CIFAR-10 and Analysis_", that contains the 
experiments conducted by the authors on the CIFAR10 dataset.

---

## :open_book: References

### :bookmark_tabs: From research papers

* [He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

### :books: From books/posts/etc.

* [Howard, J., Gugger, S., &amp; Chintala, S. (2020). Chapter 14. ResNets. In Deep learning for coders with fastai and PyTorch: AI applications without a PhD (pp. 441–458). O'Reilly Media, Inc.](https://www.amazon.es/Deep-Learning-Coders-Fastai-Pytorch/dp/1492045527)

### :computer: From code

* [Official PyTorch implementation of ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

* [PyTorch Tutorial to implement a ResNet to train with CIFAR10](https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/)

* [Un-official implementation of ResNet for CIFAR10/100 by @akamaster](https://github.com/akamaster/pytorch_resnet_cifar10)

* [Un-official implementation of ResNet for CIFAR10 by @kuanglui](https://github.com/kuangliu/pytorch-cifar)

---

## :warning: Disclaimer

All the credits go to the original authors, this is just a personal repository I create so as to
have a better understanding on the paper/s mentioned along this repository. Last but not least, 
there's a plenty of more useful resources out there, so always try to double check everything you
see (not just in this repository).