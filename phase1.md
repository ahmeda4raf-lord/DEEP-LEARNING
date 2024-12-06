
# Phase 1 - Data Collection and Preprocessing

## Overview

This repository contains the code for Phase 1 of a project aimed at emotion recognition from speech data. In this phase, the focus is on **data collection** and **preprocessing**. The dataset used is the **Toronto Emotional Speech Set (TESS)**, which consists of audio recordings with various emotions such as fear, anger, joy, sadness, disgust, and surprise. The goal of this phase is to collect, organize, and preprocess the data in preparation for subsequent phases of the project.

## What the Code Does

The notebook includes the following key steps:

1. **Library Setup and Imports:**
   The notebook starts by importing essential libraries for data manipulation, audio processing, and deep learning. This includes libraries such as:
   - `pandas`, `numpy` for data handling and manipulation
   - `seaborn`, `matplotlib` for visualization
   - `librosa`, `torchaudio` for audio data processing
   - `transformers` for future integration with pre-trained deep learning models

2. **Data Loading:**
   The **Toronto Emotional Speech Set (TESS)** dataset is loaded from the local directory. The code uses `os.walk()` to traverse the dataset's folder structure, collect the paths of all audio files, and assign the corresponding emotion labels to them.

3. **Label Distribution:**
   A simple distribution of emotion labels is generated, providing a quick overview of the dataset's balance. For instance, there are 400 files each for "fear," "angry," "disgust," "neutral," "sad," and "happy," and 200 files for "surprise."

4. **Initial Data Exploration:**
   The first few file paths and their corresponding labels are displayed, allowing for an initial verification of the dataset's structure.

5. **Preprocessing Setup:**
   Although the actual preprocessing of audio data is not fully implemented in this phase, the notebook sets up the structure for future feature extraction, including preparing the dataset for transformation into a format suitable for deep learning models.

## Key Technologies Used

- **Pandas:** For data manipulation and handling.
- **NumPy:** For numerical operations.
- **Seaborn & Matplotlib:** For data visualization.
- **Librosa & Torchaudio:** For audio signal processing.
- **PyTorch & Transformers:** For potential deep learning model integration, including models like **Wav2Vec2**.

## Achievements

- The dataset from TESS was successfully loaded and organized.
- The distribution of emotions within the dataset was analyzed and displayed.
- The preprocessing framework is ready for future steps, including feature extraction and model training.

For more details, please refer to the notebook itself:  
[Link to the notebook](https://colab.research.google.com/drive/17QLgE6x5qOoUcjrPBiXsXmx7WKaL7ElI?usp=sharing)

