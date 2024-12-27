
# Housing Price Prediction with Neural Networks

## Overview

This project aims to predict housing prices using a neural network implemented in PyTorch. The dataset includes various features related to housing characteristics, and the notebook demonstrates data preprocessing, model definition, training, and evaluation.

## Files

- **asg1(housing).ipynb**: Jupyter notebook containing the implementation of the neural network for housing price prediction. The notebook includes:
  - Importing libraries.
  - Data loading and preprocessing.
  - Neural network model definition.
  - Model training, evaluation, and performance metrics calculation.

- **Housing-1.csv**: The dataset used for training and testing the neural network. It includes 545 rows and 13 columns, representing various housing characteristics and their prices.

## Dataset

### Columns
1. `price`: Price of the house.
2. `area`: Total area of the house.
3. `bedrooms`: Number of bedrooms.
4. `bathrooms`: Number of bathrooms.
5. `stories`: Number of stories.
6. `mainroad`: Accessibility to the main road (`yes` or `no`).
7. `guestroom`: Availability of a guestroom (`yes` or `no`).
8. `basement`: Presence of a basement (`yes` or `no`).
9. `hotwaterheating`: Presence of hot water heating (`yes` or `no`).
10. `airconditioning`: Presence of air conditioning (`yes` or `no`).
11. `parking`: Number of parking spaces.
12. `prefarea`: Preference area (`yes` or `no`).
13. `furnishingstatus`: Furnishing status (`furnished`, `semi-furnished`, or `unfurnished`).

### Dataset Summary
- **Shape**: 545 rows, 13 columns.
- **Sample Data**:
  ```
  |   price    | area | bedrooms | bathrooms | stories | mainroad | guestroom | basement | ... |
  |------------|------|----------|-----------|---------|----------|-----------|----------|-----|
  | 13300000   | 7420 | 4        | 2         | 3       | yes      | no        | no       | ... |
  ```

## Notebook Structure

1. **Import Libraries**: All necessary libraries, including PyTorch and Pandas, are imported.
2. **Data Preprocessing**: Handling categorical variables, scaling features, and splitting data into training and testing sets.
3. **Define the Neural Network Model**: A custom regression model is defined using PyTorch.
4. **Train the Model**: The model is trained using mean squared error as the loss function and the Adam optimizer.
5. **Evaluate the Model**: Predictions are compared against actual values using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
6. **Visualize Results**: Outputs and evaluation metrics are visualized for better interpretation.


## How to Run

1. Install necessary libraries: `torch`, `pandas`, `sklearn`, and `matplotlib`.
2. Place the dataset (`Housing-1.csv`) in the working directory.
3. Run the Jupyter notebook (`asg1(housing).ipynb`) step by step.

