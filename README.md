# ResNet9-Model

This is the official repository for the article "to be updated soon" is to classify the images and train a model to predict any unseen images with good accuracy using `deep neural network,` `Convolutional neural network(CNN)` and `ResNet9`.

We will also use some techniques to improve our model's performance such as
1. Augmentation
2. Learning Rate Scheduling
3. gradient Clipping
4. Weight decay
5. DropOut

# Lets gets started by importing some necessary libraries
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torchvision.transforms as tt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
%matplotlib inline

# # Lets begin by downloading the Dataset
% Importing jovian python liabrary so that we will be able to commit the data of this notebook to jovian website
% project="Intel_Image_Classification_Using_ResNet9"

!pip install jovian --upgrade --quiet
% opendatasets liabrary is used here to download dataset from kaggle

!pip install opendatasets --upgrade --quiet
% import opendatasets as od
% dataset_url='add data link'
od.download(dataset_url) 
% Skipping, found downloaded files in "./intel-image-classification" (use force=True to force download)
jovian.commit()









