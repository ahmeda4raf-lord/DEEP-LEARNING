# Speech Emotion Recognition using Deep Learning

## Project Overview
This project focuses on building a deep learning framework to recognize emotions from speech signals. By leveraging advanced audio processing techniques and the Wav2Vec2 model, this project demonstrates the application of cutting-edge machine learning methods in speech emotion recognition (SER). The primary objective is to classify emotions such as happiness, sadness, anger, and fear from audio recordings with high accuracy and efficiency.

## Task Description
The main task involves developing a robust system to process audio data, extract meaningful features, and classify emotional states accurately. This is achieved using the Toronto Emotional Speech Set (TESS) dataset and the Wav2Vec2 model. The project follows these key steps:

1. **Preprocessing Audio Data:** Ensuring the dataset is clean and optimized for training.
2. **Fine-Tuning the Wav2Vec2 Model:** Adapting a pretrained model for speech emotion classification.
3. **Evaluation:** Assessing the model's performance using metrics like accuracy, precision, recall, and F1-score.

## Requirements
The project requires the following dependencies:

- Python 3.7+
- Libraries: `torch`, `transformers`, `librosa`, `matplotlib`, `seaborn`
- GPU (recommended for training efficiency)
- TESS Dataset (available on Kaggle)

## How It Works
1. **Dataset:** The TESS dataset consists of speech samples labeled with seven different emotions. This dataset is used to train and evaluate the model.
2. **Model:** The Wav2Vec2 transformer model is fine-tuned for emotion recognition by adding a classification head to predict emotion categories.
3. **Training:** The model is trained over three epochs with a batch size of 16 and a learning rate of 5e-5. Training statistics, such as runtime and loss reduction, are logged.
4. **Evaluation:** The model's performance is validated using precision, recall, and F1-score metrics. Real-world testing confirms the model's practical accuracy.

## Results
- **Training Metrics:**
  - Training Loss: 0.649
  - Samples per Second: 37.412

- **Evaluation Metrics:**
  - Loss: 0.1836
  - Accuracy: 99.81%
  - Precision: 99.81%
  - Recall: 99.81%
  - F1-Score: 99.81%

- **Real Test:** The model successfully classified 2 out of 2 real test samples.

## Future Improvements
- Expanding the dataset to include more diverse speakers and emotions.
- Incorporating ensemble methods for improved accuracy.
- Real-time deployment for use in virtual assistants and other applications.

## Authors
- Ahmed Ashraf Saad (ID: 7628)
- Mohamed Mohamed Mahmoud (ID: 7538)
