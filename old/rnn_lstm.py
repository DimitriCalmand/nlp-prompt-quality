import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from sentence_transformers import SentenceTransformer

# Parameters
n_missing = 3
MODEL_PATH = "rnn_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128

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
    
    print(f"train_contexts shape: {train_contexts.shape}")
    print(f"test_contexts shape: {test_contexts.shape}")
    
    # Filter test samples with unseen target sequences
    mask = y_test_text.isin(y_train_text.unique())
    test_contexts = test_contexts[mask]
    y_test_text = y_test_text[mask]
    print(f"Filtered test_contexts shape: {test_contexts.shape}")
    
    # Tokenize and sequence the text for RNN input
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_contexts)
    
    X_train_seq = tokenizer.texts_to_sequences(train_contexts)
    X_test_seq = tokenizer.texts_to_sequences(test_contexts)
    
    # Pad sequences to uniform length
    X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Encode target sequences
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_text)
    y_test = le.transform(y_test_text)
    
    num_classes = len(le.classes_)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Number of classes (unique missing sequences): {num_classes}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Build RNN model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    
    # Train and evaluate model
    model.fit(X_train_padded, y_train, epochs=50, batch_size=32, validation_split=0.1)
    loss, accuracy = model.evaluate(X_test_padded, y_test)
    print(f"Test Accuracy: {accuracy}")
    
    # Save the model and preprocessing objects
    model.save(MODEL_PATH)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print("Model and preprocessing objects saved.")
    
    return model, tokenizer, le

def load_objects():
    model = load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, tokenizer, le

def predict_missing_words(prompt_context, model, tokenizer, le):
    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([prompt_context])
    
    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    predicted_index = prediction.argmax(axis=1)[0]
    predicted_sequence = le.inverse_transform([predicted_index])[0]
    
    return predicted_sequence

def interactive_mode(model, tokenizer, le):
    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        user_input = input("Enter a prompt context (without the missing sequence): ")
        if user_input.lower() == "exit":
            break
        prediction = predict_missing_words(user_input, model, tokenizer, le)
        print(f"Predicted missing sequence: {prediction}\n")

def main():
    parser = argparse.ArgumentParser(description="Recurrent Neural Network Modeling for Text Sequence Prediction")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode after training or load saved model")
    parser.add_argument("--train", action="store_true", help="Force training (default loads saved model if available)")
    args = parser.parse_args()
    
    if args.train or not os.path.exists(MODEL_PATH):
        model, tokenizer, le = train_model()
    else:
        model, tokenizer, le = load_objects()
        print("Loaded saved model and preprocessing objects.")
    
    # If interactive flag is specified, enter interactive mode.
    if args.interactive:
        interactive_mode(model, tokenizer, le)
        
if __name__ == "__main__":
    main()