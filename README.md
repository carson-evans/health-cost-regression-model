# Predict Health Costs with Regression

This project utilizes machine learning to predict health insurance costs based on a dataset that includes variables like age, sex, BMI, number of children, smoking status, and geographical region. The core of this project is built using TensorFlow and Keras to implement a regression model.

## Dataset

The dataset is sourced from a publicly available dataset on [FreeCodeCamp](https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv), which includes the following columns:
- Age: age of the primary beneficiary
- Sex: insurance contractor gender, female or male
- BMI: Body mass index
- Children: Number of children covered by health insurance / Number of dependents
- Smoker: Smoking status
- Region: The beneficiary's residential area in the US
- Charges: Individual medical costs billed by health insurance

## Methodology

### Data Preparation

The dataset was cleaned and prepared for modeling. Categorical variables were encoded, and the dataset was split into training and testing sets to ensure the model's ability to generalize well to new data.

### Exploratory Data Analysis (EDA)

We performed an exploratory analysis to understand the distribution of different variables and their relationship with the insurance costs.

### Model Building

A neural network model was constructed using TensorFlow and Keras, with the following architecture:
- Input Normalization Layer
- Dense Layer with 64 units and ReLU activation
- Another Dense Layer with 64 units and ReLU activation
- Output Layer with a single unit

The model was compiled with the Adam optimizer and mean absolute error as the loss function.

### Training

The model was trained on the training dataset, with validation split to monitor and prevent overfitting.

## Results

After training, the model achieved a mean absolute error (MAE) on the test dataset. The MAE indicates the average difference between the predicted insurance costs and the actual costs in the test data.

## Usage

To run this project, you will need to install certain Python libraries. You can install the dependencies using the following command:

```bash
pip install -q git+https://github.com/tensorflow/docs
```

And then import the necessary libraries in your Python environment:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
```

Follow the steps in the Jupyter Notebook to proceed with data loading, preprocessing, model building, training, and evaluation.

## Conclusion

This project demonstrates the application of neural networks to predict health insurance costs. The model provides a foundational approach that can be further refined and improved with more complex architectures or feature engineering.
