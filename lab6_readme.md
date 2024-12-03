# Image Captioning with ResNet50 and Transformers (RNN) - Deep Learning Project

This repository contains a deep learning project that generates **image captions** using a combination of **ResNet50** (a pre-trained Convolutional Neural Network) and **Transformers (RNN)**. The goal of this project is to develop a model capable of generating human-readable descriptions of images, which is a key task in computer vision and natural language processing.

## Project Overview

In this project, we combine the power of **ResNet50** for feature extraction from images with **Transformers (RNN)** to generate captions. The deep learning model works as follows:

1. **ResNet50**: This pre-trained CNN model is used to extract meaningful features from images. The features from the last convolutional layer are fed into the RNN.
   
2. **Transformer (RNN)**: The extracted image features are then passed to a Recurrent Neural Network (RNN), which processes the features and generates a sequence of words to form a coherent caption.

### Key Components:
- **ResNet50** for image feature extraction.
- **Transformer (RNN)** for caption generation.
- **Flickr8k dataset** for training and testing the model.

## Notebook Link

You can access the project notebook here:

[Image Captioning with ResNet50 and Transformers - Colab Notebook](https://colab.research.google.com/drive/1qAEb808UpqdVRFxSXhvngXka67qKfuHz?usp=sharing)

## Libraries and Dependencies

This project requires the following Python libraries for training the model and processing the data:

- **TensorFlow / Keras** for building and training the deep learning model.
- **NumPy** for numerical operations.
- **Matplotlib** for visualizations.
- **OpenCV** for image processing tasks.



