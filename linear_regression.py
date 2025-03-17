import pandas as pd

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_parquet("dataset.parquet")

df = df[df["num_responses"] > 1]
df = df[df["agreement_ratio"] > 0.4]
df["binary_grade"] = df["kind"].apply(lambda x: 1 if x == "human" else 0)


X = df["prompt"]
y = df["binary_grade"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")  # Lightweight and fast

X_train = model.encode(X_train.tolist())
X_test = model.encode(X_test.tolist())

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 100,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

prompt = """I love bananas, can you make a recipe out of it ?"""

prompt_vector = model.encode([prompt])
prediction = lr.predict(prompt_vector)
print(f"Prediction: {prediction}")

