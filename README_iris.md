
# Iris Dataset Classification Project

## Objective
The objective of this project is to build a neural network model using PyTorch to classify flowers from the Iris dataset into three species: Setosa, Versicolour, and Virginica. The task involves loading the dataset, preprocessing, building a deep learning model, training it, and evaluating its performance.

## Dataset Description
The Iris dataset is a well-known dataset in machine learning and statistics. It consists of 150 samples from three species of Iris flowers, with four features for each sample:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target variable indicates the species of the flower. This dataset is widely used as a beginner's introduction to classification problems.

## Steps to Run the Code in Jupyter
1. **Clone or Download**:
   - Clone this repository or download the project files.
2. **Install Python**:
   - Ensure Python 3.7+ is installed on your system.
3. **Install Dependencies**:
   - Use the following command to install all required dependencies:
     ```bash
     pip install pandas torch matplotlib
     ```
4. **Run the Notebook**:
   - Open the notebook `iris_7628.ipynb` in Jupyter Notebook or Jupyter Lab.
   - Execute each cell step by step to load the dataset, preprocess it, build the model, train it, and evaluate its performance.

## Dependencies and Installation Instructions
Install the following Python libraries to run the notebook:
- **pandas**: For data manipulation and analysis.
- **torch**: For building and training neural network models.
- **matplotlib**: For visualizing data and model performance.

Install them using:
```bash
pip install pandas torch matplotlib
```

## Project Discussion
### Highlights:
1. **Deep Learning with PyTorch**:
   - The project leverages PyTorch to build and train a simple feedforward neural network for classification.
   - PyTorch is highly flexible and efficient for custom model creation and training.

2. **Dataset Preprocessing**:
   - The dataset is loaded using pandas and normalized to improve model performance.

3. **Model Architecture**:
   - The neural network consists of an input layer, one or more hidden layers, and an output layer with three neurons (for the three classes).
   - The activation function used is ReLU for hidden layers and Softmax for the output.

4. **Training and Evaluation**:
   - The model is trained using the Cross-Entropy loss function and evaluated for accuracy and loss on the test set.
   - Visualizations of the loss and accuracy trends during training are included.

5. **Challenges and Benefits**:
   - The Iris dataset is relatively small, making it suitable for learning classification techniques but not for large-scale deployments.
   - Despite its simplicity, the project demonstrates the power of neural networks for solving classification problems.
