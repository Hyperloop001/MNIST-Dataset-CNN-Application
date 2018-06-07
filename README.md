# MNIST Dataset Implemented with CNN 

## This is what it should be like if you succeed:
![alt text](https://i.imgur.com/lup5gsA.png)
![alt text](https://i.imgur.com/XPMA0A8.png)

## Before you download it, check these out:
    This module contains:
        1. mnistAppBasicLinear.py: Basic simplified algorithm for training and utilizing the MNIST dataset.   
        2. mnistAppBetterCNN.py: Better implementation of a Convolutional-Neural-Network designed for training and utilizing the MNIST dataset.     
        3. mnistFunc.py: Graphic and algorithm supports to the above implementations.

## If you wanna play with it, follow these instructions:
    Train the model by yourself:
        1. Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/
        2. Place the .gz files under MNIST_data\trainData
        3. Set "train = True" in mnistAppBasicLinear.py or mnistAppBetterCNN.py, depending on the algorithm you are using
        4. Modify the model storing path to store a new version
        Note: Pre-trained model are stored within MNIST_model\basic and MNIST_model\better            
    Test with hand written digits:
        1. Set "train = False" and "useHandWrittenDigit = True"
        2. Hit run, if images were read in correctly, each digit and its weight distribution should be displayed
        Note: Image for hand written digits are placed under MNIST_data\applicationData, you can creat your own as well
        
        
