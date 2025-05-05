import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
import joblib
from sentence_transformers import SentenceTransformer

# Attention c'est un single layer pas multi

# Parameters
n_missing = 3
MODEL_PATH = "nn_model.h5"
VECTORIZER_PATH = "vectorizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

def train_model():
    # Load dataset
    df = pd.read_parquet("hf://datasets/data-is-better-together/10k_prompts_ranked/data/train-00000-of-00001.parquet")
    df['prompt'] = df.prompt.astype(str)
    
    # Target and context creation functions
    def get_last_n_words(text, n=n_missing):
        words = text.split()
        if len(words) >= n:
            return " ".join(words[-n:]).lower()
        else:
            return " ".join(words).lower()
    
    def get_context_n(text, n=n_missing):
        words = text.split()
        return " ".join(words[:-n]) if len(words) > n else ""
    
    # Create target and context columns
    df['missing_words'] = df['prompt'].apply(get_last_n_words)
    df['prompt_context'] = df['prompt'].apply(get_context_n)
    
    # Split data (80/20)
    split_index = int(df.shape[0] * 0.8)
    train_contexts = df['prompt_context'][:split_index]
    test_contexts = df['prompt_context'][split_index:]
    y_train_text = df['missing_words'][:split_index]
    y_test_text = df['missing_words'][split_index:]
    
    print(f"train_contexts shape : {train_contexts.shape}")
    print(f"test_contexts shape : {test_contexts.shape}")
    
    # Filter test samples with unseen target sequences
    mask = y_test_text.isin(y_train_text.unique())
    test_contexts = test_contexts[mask]
    y_test_text = y_test_text[mask]
    print(f"Filtered test_contexts shape : {test_contexts.shape}")
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(train_contexts)
    X_test_tfidf = vectorizer.transform(test_contexts)
    
    # Load the sentence transformer model for embeddings
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    train_embeddings = embed_model.encode(train_contexts.tolist())
    test_embeddings = embed_model.encode(test_contexts.tolist())
    
    # Combine TF-IDF and embeddings
    X_train_combined = np.hstack([X_train_tfidf.toarray(), train_embeddings])
    X_test_combined = np.hstack([X_test_tfidf.toarray(), test_embeddings])
    
    # Encode target sequences
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_text)
    y_test = le.transform(y_test_text)
    
    num_classes = len(le.classes_)
    print(f"Number of classes (unique missing sequences): {num_classes}")
    
    # Build neural network model
    input_dim = X_train_combined.shape[1]
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train and evaluate model
    model.fit(X_train_combined, y_train, epochs=10, batch_size=32, validation_split=0.1)
    loss, accuracy = model.evaluate(X_test_combined, y_test)
    print(f"Test Accuracy: {accuracy}")
    
    # Save the model and preprocessing objects
    model.save(MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print("Model and preprocessing objects saved.")
    
    return model, vectorizer, le, embed_model

def load_objects():
    model = load_model(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    # Load sentence transformer model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, vectorizer, le, embed_model

def predict_missing_words(prompt_context, model, vectorizer, le, embed_model):
    # Compute TF-IDF vector and sentence embedding
    tfidf_vec = vectorizer.transform([prompt_context]).toarray()
    embed_vec = embed_model.encode([prompt_context])
    combined_vec = np.hstack([tfidf_vec, embed_vec])
    prediction = model.predict(combined_vec)
    predicted_index = prediction.argmax(axis=1)[0]
    predicted_sequence = le.inverse_transform([predicted_index])[0]
    return predicted_sequence

def interactive_mode(model, vectorizer, le, embed_model):
    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        user_input = input("Enter a prompt context (without the missing sequence): ")
        if user_input.lower() == "exit":
            break
        prediction = predict_missing_words(user_input, model, vectorizer, le, embed_model)
        print(f"Predicted missing sequence: {prediction}\n")

def main():
    parser = argparse.ArgumentParser(description="Feed Forward NN Modeling with TF-IDF and Embeddings")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode after training or load saved model")
    parser.add_argument("--train", action="store_true", help="Force training (default loads saved model if available)")
    args = parser.parse_args()
    
    if args.train or not os.path.exists(MODEL_PATH):
        model, vectorizer, le, embed_model = train_model()
    else:
        model, vectorizer, le, embed_model = load_objects()
        print("Loaded saved model and preprocessing objects.")
    
    # If interactive flag is specified, enter interactive mode.
    if args.interactive:
        interactive_mode(model, vectorizer, le, embed_model)
        
if __name__ == "__main__":
    main()