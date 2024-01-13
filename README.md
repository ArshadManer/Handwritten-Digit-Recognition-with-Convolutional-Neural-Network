# Handwritten Digit Recognition with Convolutional Neural Network

## Overview
This project implements a Convolutional Neural Network (ConvNet) for handwritten digit recognition using the MNIST dataset. The model is built using Keras, a high-level deep learning API, with TensorFlow as the backend for low-level operations. The project also utilizes NumPy for matrix operations and Matplotlib for visualization.

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits. The images have been normalized and centered, making it a widely used benchmark dataset for digit recognition tasks. The dataset is easily accessible through the TensorFlow dataset API, streamlining the data loading process.

## Project Structure
The project is organized into a Python class named MNISTClassifier, encapsulating the functionality for loading data, preprocessing, building, training, and evaluating the ConvNet. The class uses Keras for model construction and TensorFlow for backend operations.

## Components of ConvNets
Conv2D Layer: Utilized for feature extraction through convolution.

Filters: Number of feature detectors.
Kernel Size: Shape of the feature detector.
Strides: Control how many units the filter shifts.
Padding: Optional argument controlling dimensionality (valid or same).
Pooling Layer: Max pooling used to downsample the spatial dimensions.

Pool Size: Shape of the pooling window.
Fully Connected Layer (Dense Layer): Classification layer for making predictions.

Dropout Layer: Prevents overfitting by randomly dropping activations.

## Results
The ConvNet achieved an impressive accuracy of 98% on the test dataset, showcasing its effectiveness in handwritten digit recognition.