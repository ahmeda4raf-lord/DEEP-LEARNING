
## Notebook Link

You can access the project notebook here:

[Image Captioning with ResNet50 and Transformers - Colab Notebook](https://colab.research.google.com/drive/1qAEb808UpqdVRFxSXhvngXka67qKfuHz?usp=sharing)




# Image Captioning using ResNet50

This repository contains a deep learning model for generating captions for images using a combination of ResNet50 for image feature extraction and an LSTM-based model for text generation.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup Instructions](#setup-instructions)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Model Implementation](#model-implementation)
5. [Training the Model](#training-the-model)
6. [Real-Time Caption Generation](#real-time-caption-generation)

## Introduction

The notebook implements an image captioning solution by combining the power of ResNet50 for feature extraction and LSTM networks for caption generation. The dataset used is the Flickr8k dataset, which contains images and their corresponding captions. 

## Setup Instructions

### 1. Installing Dependencies

Ensure you have the required libraries installed. You can install them using:

```bash
pip install tensorflow numpy tqdm keras
```

### 2. Download Dataset

You can download the Flickr8k dataset using the Kaggle API. This dataset is used for training and testing the model.

```python
import kagglehub
path = kagglehub.dataset_download('adityajn105/flickr8k')
print('Path to dataset files:', path)
```

Alternatively, you can manually download the dataset and place it in the appropriate folder.

### 3. Import Required Libraries

In the initial step, all necessary libraries are imported for model training, image preprocessing, and text tokenization.

```python
import os
import numpy as np
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from keras.utils import to_categorical
from tensorflow.keras import layers
```

## Preprocessing Steps

### 1. Loading Images and Extracting Features

The dataset is loaded and the features of each image are extracted using the ResNet50 model. These features are then saved for later use.

```python
# Load the ResNet50 model without the top layer
resnet50 = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    pooling='avg'
)

# Preprocess and extract features from each image in the dataset
directory = '/content/flickr8k/Images'
features = {}

for img_name in tqdm(os.listdir(directory)):
    img_path = os.path.join(directory, img_name)
    image = load_img(img_path, target_size=(224, 224))  # Resize image to match ResNet input
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)  # Preprocess image for ResNet50
    feature = resnet50.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature
```

### 2. Building the Captions Dictionary

Captions are loaded and mapped to their respective image IDs. The dictionary stores this information for later model training.

```python
# Load captions from file and map to image IDs
with open('captions.txt') as f:
    captions_doc = f.read()

mapping = {}
for line in tqdm(captions_doc.split('
')):
    tokens = line.split(',')
    if len(tokens) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)
```

### 3. Tokenizing the Captions

The captions are tokenized and padded so that they can be used as input for the LSTM model.

```python
# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

# Pad sequences to a fixed length
max_sequence_length = 34
sequences = tokenizer.texts_to_sequences(captions)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
```

## Model Implementation

### 1. Building the Model

The model consists of two main parts:
- **Feature Extractor**: Uses ResNet50 to extract image features.
- **Text Generator**: Uses LSTM layers to generate captions.

```python
# Define model architecture
input_image_features = layers.Input(shape=(2048,))
image_features = layers.Dense(256, activation='relu')(input_image_features)

input_sequence = layers.Input(shape=(max_sequence_length,))
caption_sequence = layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=256)(input_sequence)
caption_sequence = layers.LSTM(256)(caption_sequence)

decoder_input = layers.Add()([image_features, caption_sequence])
decoder_output = layers.Dense(len(tokenizer.word_index)+1, activation='softmax')(decoder_input)

model = Model(inputs=[input_image_features, input_sequence], outputs=decoder_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Training the Model

### 1. Training the Model

The model is trained using the features extracted from images and their corresponding captions.

```python
model.fit([features, padded_sequences], to_categorical(labels), epochs=10, batch_size=32)
```

### 2. Saving the Model

After training, the model is saved for future use.

```python
model.save('image_caption_model.h5')
```

## Real-Time Caption Generation

Once the model is trained, it can generate captions for unseen images by extracting features and predicting captions.

```python
def generate_caption(model, image_path):
    # Extract features and predict caption
    image_features = extract_image_features(image_path)
    predicted_caption = predict_caption(model, image_features)
    return predicted_caption
```




