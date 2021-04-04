## :test_tube: ResNet-20 Implementation

In order to understand how does the ResNet architecture works, we will be implementing the simplest version of it, which
is the ResNet20 for CIFAR10. This exercise will be useful to understand the main differences between a plain convolutional
neural network and deep residual network, and the functionality of the 
"_shortcut connections_" in the residual building blocks.

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

* The convolutional filters to be applied are 16, 32, and 64; so that the size of the input image goes 
from 32x32 to 16x16, and then to 8x8.

* The neural network starts off with a convolutional layer which applies a 3x3 convolution, resulting in 
16 out channels.

* Then we will include the residual blocks (the basic, not the bottleneck ones), that is a stack of `6*n`
layers with 3x3 convolutions, that contains `2n` layers for each feature map size.

* The subsampling/downsampling is performed by convolutions with a stride of 2, instead of using pooling operations. 
A comparison between both approaches can be found at: 
[Stackexchange: Pooling vs. stride for downsampling](https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling/387522)

* Finally, the neural network ends with a global average pooling and a fully connected linear layer with 10 units
that stands for the 10 classes of the CIFAR10 dataset.