# MNIST Digit Classification Project

This project demonstrates the use of various machine learning models to classify handwritten digits from the MNIST dataset. The models used in this project include Convolutional Neural Networks (CNN), Deep Neural Networks (DNN), Logistic Regression, and Random Forest.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models Used](#models-used)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [Deep Neural Network (DNN)](#deep-neural-network-dnn)
  - [Logistic Regression](#logistic-regression)
  - [Random Forest](#random-forest)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)


## Introduction

The goal of this project is to accurately classify handwritten digits (0-9) using the MNIST dataset. This dataset is widely used for training and testing in the field of machine learning and consists of 60,000 training images and 10,000 test images.

## Dataset

The MNIST dataset contains grayscale images of handwritten digits, each of size 28x28 pixels. Each image is associated with a label indicating the digit it represents.

## Models Used

### Convolutional Neural Network (CNN)

A CNN is a class of deep neural networks that is particularly effective for image recognition tasks. It consists of convolutional layers that automatically and adaptively learn spatial hierarchies of features from input images.

### Deep Neural Network (DNN)

A DNN is a neural network with multiple hidden layers between the input and output layers. These networks are capable of modeling complex patterns in data.

### Logistic Regression

Logistic Regression is a simple yet effective model for binary and multiclass classification problems. It models the probability that a given input belongs to a particular class.

### Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes as the prediction.

## Results

The performance of each model was evaluated using accuracy, precision, recall, and F1-score metrics. Below is a summary of the results:

- **CNN**:
  - Accuracy: 0.988000
  - Precision: 0.988065
  - Recall: 0.988000
  - F1-score: 0.988006

- **DNN**:
  - Accuracy: 0.925800
  - Precision: 0.926027
  - Recall: 0.925800
  - F1-score: 0.925755

- **Logistic Regression**:
  - Accuracy: 0.920800
  - Precision: 0.920617
  - Recall: 0.920800
  - F1-score: 0.920641

- **Random Forest**:
  - Accuracy: 0.965500
  - Precision:0.965572
  - Recall: 0.965500
  - F1-score: 0.965505

## Conclusion

The project demonstrates the effectiveness of different machine learning models for the task of handwritten digit classification. The CNN model, with its ability to capture spatial hierarchies in images, performed the best among the models tested.

## Installation

To run this project, you'll need to have Python and the following libraries installed:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- seaborn

You can install these libraries using pip:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn seaborn
