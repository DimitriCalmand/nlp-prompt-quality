import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_splitcd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TOKENIZER_TYPE = 'BOW'  # Choose from 'BOW', 'TF-IDF', 'TRANSFORMER'
LABEL_TYPE = 'KIND'  # Choose from 'KIND', 'GRADE'
PROMPT = """I love bananas, can you make a recipe out of it ?"""


##########################
### DATA PREPROCESSING ###
##########################

df = pd.read_parquet("dataset.parquet")

#df = df[df["num_responses"] > 1]
df = df[df["agreement_ratio"] > 0.4]

if LABEL_TYPE == 'KIND':
    df["label_binary"] = df["kind"].apply(lambda x: 1 if x == "human" else 0)
elif LABEL_TYPE == 'GRADE':
    df["label_binary"] = df["avg_rating"].apply(lambda x: 1 if x >= 4 else 0)
else:
    raise ValueError("Invalid LABEL_TYPE. Choose from 'KIND', 'GRADE'")

X = df["prompt"]
y = df["label_binary"]


########################
### TOKENIZE PROMPTS ###
########################

# BAG OF WORDS
if TOKENIZER_TYPE == 'BOW':
    tokenizer = CountVectorizer()
    X = tokenizer.fit_transform(X)

# TF-IDF
elif TOKENIZER_TYPE == 'TF-IDF':
    tokenizer = TfidfVectorizer()
    X = tokenizer.fit_transform(X)

# TRANSFORMER-BASED SENTENCE EMBEDDING
elif TOKENIZER_TYPE == 'TRANSFORMER':
    tokenizer = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")  # Lightweight and fast
    X = tokenizer.encode(X.tolist())

else:
    raise ValueError("Invalid TOKENIZER_TYPE. Choose from 'BOW', 'TF-IDF', 'TRANSFORMER'")

######################
### TRAINING MODEL ###
######################

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

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


######################
### EVALUATE MODEL ###
######################

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


#######################
### MAKE PREDICTION ###
#######################

if TOKENIZER_TYPE == 'BOW' or TOKENIZER_TYPE == 'TF-IDF':
    prompt_vector = tokenizer.transform([PROMPT])
elif TOKENIZER_TYPE == 'TRANSFORMER':
    prompt_vector = tokenizer.encode([PROMPT])
else:
    raise ValueError("Invalid TOKENIZER_TYPE. Choose from 'BOW', 'TF-IDF', 'TRANSFORMER'")

print(f"\nPrompt: {PROMPT}")
prediction = lr.predict(prompt_vector)
print(f"Prediction: {prediction[0]}")
