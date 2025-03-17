import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

# Charger votre jeu de données (assurez-vous qu'il possède les colonnes 'prompt' et 'avg_rating')
data = pd.read_parquet("hf://datasets/data-is-better-together/10k_prompts_ranked/data/train-00000-of-00001.parquet")

# Préparer les features (embeddings du prompt) et la cible (la note entre 0 et 5)
model_embed = SentenceTransformer("paraphrase-MiniLM-L6-v2")

X = np.array([model_embed.encode(prompt) for prompt in data['prompt']])
y = np.round(data['avg_rating']).astype(int)

# Diviser les données en un jeu d'entraînement et un jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle naïf bayésien
model = GaussianNB()  # Naive Bayes gaussien pour les features continues
model.fit(X_train, y_train)

# Prédire les notes du jeu de test
y_pred = model.predict(X_test)

# Évaluer le modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Sauvegarder le modèle
joblib.dump(model, "naive_bayes_model.pkl")