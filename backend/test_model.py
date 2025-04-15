import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.pipeline import Pipeline


# Download NLTK resources if needed
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

def preprocess_text(text):
    """Preprocess text for TF-IDF vectorization"""
    if text is None or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Model paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_svm_tfidf.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Test data - various Indonesian comments
test_comments = [
    "Pelayanan parah banget, nyesel beli di sini!",                   
    "Gak nyangka, produknya lebih jelek dari ekspektasi!",             
    "B aja sih, gak istimewa tapi juga gak jelek banget.",           
    "Luar biasa! Pelayanan super ramah dan cepet banget!",             
    "Wah, ini sih produk terbaik yang pernah gue beli!",              
    "Produknya standar, sesuai harga lah ya, gak ada yang spesial.",
    "Presiden Anjing",
    "Biasa aja ah"
]

print("Testing SVM model with sample comments")
print("======================================")

try:
        # Try to load model
        # Try to load model
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    if isinstance(model, Pipeline):
        print("Model is a pipeline with built-in vectorizer")

        preprocessed_comments = [preprocess_text(comment) for comment in test_comments]

        predictions = model.predict(preprocessed_comments)
        try:
            probabilities = model.predict_proba(preprocessed_comments)
        except:
            print("Model does not support predict_proba, using decision_function instead")
            probabilities = model.decision_function(preprocessed_comments)

        label_map = {0: "negative", 1: "neutral", 2: "positive"}

        print("\nPrediction Results:")
        print("-------------------")
        for i, (comment, pred, prob) in enumerate(zip(test_comments, predictions, probabilities)):
            sentiment = label_map.get(pred, "unknown")
            print(f"{i+1}. Text: \"{comment}\"")
            print(f"   Prediction: {pred} ({sentiment})")
            print(f"   Confidence/Raw Scores: {prob}")
            print()
    else:
        print("Model is not a pipeline. Please provide vectorizer separately.")

        
        # Try to load vectorizer
        if os.path.exists(VECTORIZER_PATH):
            print(f"Loading vectorizer from {VECTORIZER_PATH}")
            vectorizer = joblib.load(VECTORIZER_PATH)
            
            # Preprocess comments
            preprocessed_comments = [preprocess_text(comment) for comment in test_comments]
            
            # Transform using vectorizer
            features = vectorizer.transform(preprocessed_comments)
            
            # Make predictions
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            
            # Define label mapping
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            
            # Print results
            print("\nPrediction Results:")
            print("-------------------")
            for i, (comment, pred, prob) in enumerate(zip(test_comments, predictions, probabilities)):
                sentiment = label_map.get(pred, "unknown")
                print(f"{i+1}. Text: \"{comment}\"")
                print(f"   Prediction: {pred} ({sentiment})")
                print(f"   Probabilities: {prob}")
                print()
        else:
            print(f"Error: Vectorizer not found at {VECTORIZER_PATH}")
            print("Cannot test the model without a vectorizer")

except Exception as e:
    print(f"Error testing model: {str(e)}")
    
print("\nPenting: Pastikan model dan vectorizer tersedia di direktori models/")
print("Jika menggunakan pipeline, hanya file best_model_svm_tfidf.pkl yang diperlukan")
print("Jika model terpisah, kedua file best_model_svm_tfidf.pkl dan tfidf_vectorizer.pkl diperlukan") 