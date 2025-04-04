# NLP Project: Prompt Evaluation

# Linear Regression
The `linear_regression.py` file performs linear regression on a dataset. It uses the pandas library to load and preprocess the data, and the scikit-learn library to train and evaluate a linear regression model. The script includes the following main steps:  

## How to Use
**Set Parameter:**  
Adjust the `TOKENIZER_TYPE`  variable to choose the tokenization method.
Adjust the `PROMPT` variable to run a prediction a specific prompt.

**Run the Script:**  
Execute the script from the command line or an IDE to load the dataset, train the model, evaluate it, and make a prediction on a sample `PROMPT`.

**View Results:**  
The script will print the dataset information, model evaluation metrics, and the prediction for the sample prompt.


### 1. Data Loading and Preprocessing:
- Loads a dataset from a Parquet file.
- Filters the dataset based on specific conditions. 

### 2. Tokenization:  
Tokenizes the text data using one of three methods:
- Bag of Words ("BOW")
- TF-IDF
- Transformer-based sentence embeddings ("TRANSFORMER")

### 3. Model Training:  
Splits the data into training and test sets.
Trains a linear regression model using the training data.

### 4. Model Evaluation: 
Evaluates the model on the test set and prints **Mean Squared Error and R^2 Score**.
Makes a prediction on a sample prompt using the trained model and prints the result.
To run the script, use the following command: `python linear_regression.py`


# Logistic Regression
The `logistic_regression.py` file performs logistic regression on a dataset. It uses the pandas library to load and preprocess the data, and the scikit-learn library to train and evaluate a logistic regression model. The script includes the following main steps:  

## How to Use
**Set Parameters:**  
Adjust the `TOKENIZER_TYPE` and `LABEL_TYPE` variables to choose the tokenization method and label type.
Adjust the `PROMPT` variable to run a prediction a specific prompt.

**Run the Script:**  
Execute the script from the command line or an IDE to load the dataset, train the model, evaluate it, and make a prediction on a sample `PROMPT`.

**View Results:**  
The script will print the dataset information, model evaluation metrics, and the prediction for the sample prompt.


### 1. Data Loading and Preprocessing:
- Loads a dataset from a Parquet file.
- Filters the dataset based on specific conditions. 
- Creates binary labels based on the `LABEL_TYPE` parameter.

### 2. Tokenization:  
Tokenizes the text data using one of three methods:
- Bag of Words ("BOW")
- TF-IDF
- Transformer-based sentence embeddings ("TRANSFORMER")

### 3. Model Training:  
Splits the data into training and test sets.
Trains a logistic regression model using the training data.

### 4. Model Evaluation: 
Evaluates the model on the test set and prints **accuracy, precision, recall, and F1 score**.
Makes a prediction on a sample prompt using the trained model and prints the result.
To run the script, use the following command: