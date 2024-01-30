## ResNet9 - PyTorch

We'll implement a significantly larger model this time, called the ResNet9, which has 9 convolutional layers. However, one of the key changes to our model is the addition of the **residual block**, which adds the original input back to the output feature map obtained by passing the input through one or more convolutional layers.
![ResNet9 Architecture](txt-Defined ResNet-9)

Here, we're trying to build a deep residual neural network to classify images from the CIFAR10 dataset with around 90+ accuracy. In this project, we'll use the following techniques to achieve SOTA accuracy:

-   Data normalization
-   Data augmentation
-   Residual connections
-   Batch normalization
-   Learning rate annealing
-   Weight Decay

### Technologies Used
- PyTorch 1.5.x

# Contact
If you need any help with the code below, contact me at wasswashafik [at] dcrlab [dot] org
