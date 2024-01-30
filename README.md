## ResNet9 - PyTorch

We'll use a significatly larger model this time, called the ResNet9, which has 9 convolutional layers. However, one of the key changes to our model is the addition of the **resudial block**, which adds the original input back to the output feature map obtained by passing the input through one or more convolutional layers.
![ResNet9 Architecture](txt-Defined ResNet-9)

Here we're trying to build a deep residual neural network to classify images from the CIFAR10 dataset with around 90+ accuracy. In this project, we'll use the following techniques to achieve SOTA accuracy in less than 10 minutes:

-   Data normalization
-   Data augmentation
-   Residual connections
-   Batch normalization
-   Learning rate annealing
-   Weight Decay

### Technologies Used
- PyTorch 1.5.x

### Training

    python train.py

### Reference
1. ![http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf)
2. ![https://github.com/apple/ml-cifar-10-faster](https://github.com/apple/ml-cifar-10-faster)
