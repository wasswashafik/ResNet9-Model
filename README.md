# ResNet9-Model - PyTorch

This is the official repository for the article "to be updated soon" is to classify the images and train a model to predict any unseen images with good accuracy using `deep neural network,` `Convolutional neural network(CNN)` and `ResNet9`.

We'll implement a significantly larger model this time, called the ResNet, which has 9 convolutional layers. However, one of the key changes to our model is the addition of the **residual block**, which adds the original input back to the output feature map obtained by passing the input through one or more convolutional layers.
![ResNet9 Architecture](txt-Defined ResNet-9)

Here, we're trying to build a deep residual neural network to classify images from the TADD dataset with around 90+ accuracy. In this project, we'll use the following techniques to achieve SOTA accuracy in a few minutes:

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






