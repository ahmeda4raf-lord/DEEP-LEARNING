
# Orbit Trajectory Prediction with Neural Networks

## Overview

This project predicts the trajectory of an orbit using a neural network implemented in TensorFlow and Keras. The dataset consists of time steps and corresponding vertical positions. The notebook demonstrates data loading, preprocessing, model definition, training, and prediction visualization.

## Files

- **asg1(orbit).ipynb**: Jupyter notebook containing the implementation of the neural network for orbit trajectory prediction. The notebook includes:
  - Data loading and preprocessing.
  - Neural network model definition.
  - Model training and evaluation.
  - Visualization of predicted trajectories.

- **orbit.csv**: The dataset used for training and testing the neural network. It includes 2,000 rows and 2 columns, representing time steps and vertical positions.

## Dataset

### Columns
1. `time_steps`: The time steps for the orbit trajectory.
2. `y`: The vertical positions corresponding to each time step.

### Dataset Summary
- **Shape**: 2,000 rows, 2 columns.
- **Sample Data**:
  ```
  | time_steps   | y                  |
  |--------------|--------------------|
  | -10.0        | 100.0             |
  | -9.989995    | 99.80000005005004 |
  | -9.979990    | 99.60020030025018 |
  ```

## Notebook Structure

1. **Import Libraries**: Importing TensorFlow, NumPy, Pandas, and necessary libraries.
2. **Load Data**: Loading the dataset and confirming successful data import.
3. **Preprocessing**: Converting data to NumPy arrays for training, applying data augmentation.
4. **Define the Neural Network Model**: Building a sequential neural network with customizable layers and dropout.
5. **Compile the Model**: Using the Adam optimizer and Mean Squared Error (MSE) loss function.
6. **Train and Evaluate**: Training the model and visualizing the training performance.
7. **Prediction and Visualization**: Comparing the predicted orbit trajectory with the original scientist-provided trajectory.

## How to Run

1. Install necessary libraries: `tensorflow`, `keras`, `numpy`, `pandas`, and `matplotlib`.
2. Place the dataset (`orbit.csv`) in the working directory.
3. Run the Jupyter notebook (`asg1(orbit).ipynb`) step by step.
