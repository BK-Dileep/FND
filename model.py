# model.py

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import nltk
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels to the datasets
df_fake['label'] = 0  # Fake news
df_true['label'] = 1  # Real news

# Combine the datasets
df_combined = pd.concat([df_fake, df_true], ignore_index=True)

# Text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Apply text preprocessing
df_combined['text'] = df_combined['text'].apply(preprocess_text)

# Split the data into features and labels
X = df_combined['text']
y = df_combined['label']

# Tokenize text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform length
X_padded = pad_sequences(X_sequences, maxlen=100, padding='post')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.3, random_state=42)

# Build the RNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')
])

# Compile the RNN model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the RNN model
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

# Use the trained RNN model to predict probabilities
train_probs = model.predict(X_train)
test_probs = model.predict(X_test)

# Train a Gradient Boosting Classifier on the RNN probabilities
gb_model = GradientBoostingClassifier()
gb_model.fit(train_probs, y_train)

# Save the model
joblib.dump(gb_model, 'gb_model.pkl')

# Function to preprocess text and predict fake news
# Function to preprocess text and predict fake news
def predict_fake_news(text):
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    # Tokenize text
    X_sequence = tokenizer.texts_to_sequences([preprocessed_text])
    # Pad sequence
    X_padded = pad_sequences(X_sequence, maxlen=100, padding='post')
    # Get the RNN model prediction
    rnn_prediction = model.predict(X_padded)
    # Reshape to match classifier input
    rnn_prediction_reshaped = rnn_prediction.reshape(-1, 1)
    # Predict using Gradient Boosting Classifier
    prediction = gb_model.predict(rnn_prediction_reshaped)[0]
    return prediction
