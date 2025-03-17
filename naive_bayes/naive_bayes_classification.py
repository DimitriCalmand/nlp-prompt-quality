import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# Charger le modèle Naïf Bayésien sauvegardé
model_nb = joblib.load("naive_bayes_model.joblib")

# Charger le modèle d'embeddings (le même utilisé lors de l'entraînement)
model_embed = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Exemple de prompts à classifier
exemples_prompts = [
    "Banana",
    "2+2 = ? ",
    "Make me a simple code in Python that prints 'Hello, World!'",
    "Provide step-by-step instructions on how to make a safe and effective homemade all-purpose cleaner from uncommon household ingredients. The guide should not include measurements, tips for storing the cleaner, and additional variations or scents that can be added. It should be written in clear language, with useful visuals or photographs to aid in the process."

]


# Générer les embeddings pour les prompts
X_exemples = np.array([model_embed.encode(prompt) for prompt in exemples_prompts])

# Réaliser la classification avec le modèle chargé
predictions = model_nb.predict(X_exemples)

# Afficher les résultats
for prompt, pred in zip(exemples_prompts, predictions):
    print(f"Prompt : {prompt}")
    print(f"Classe prédite : {pred}")
    print("-----")
