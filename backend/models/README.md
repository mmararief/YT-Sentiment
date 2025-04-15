# Model Machine Learning

Direktori ini berisi model-model machine learning yang digunakan untuk analisis sentimen komentar YouTube.

## Model yang Tersedia

### 1. SVM dengan TF-IDF Vectorizer

File-file model:

- `best_model_svm_tfidf.pkl` - Model SVM yang sudah dilatih
- `tfidf_vectorizer.pkl` - TF-IDF Vectorizer yang digunakan untuk mengubah teks menjadi fitur

## Tentang Model

Model SVM (Support Vector Machine) dengan TF-IDF (Term Frequency-Inverse Document Frequency) adalah pendekatan klasik namun efektif untuk klasifikasi teks. Model ini dilatih dengan dataset komentar YouTube dalam bahasa Indonesia dan mampu mengklasifikasikan sentimen menjadi 3 kategori:

- Positif (2)
- Netral (1)
- Negatif (0)

## Menggunakan Model

Model ini digunakan secara otomatis oleh aplikasi ketika pengguna memilih opsi "SVM with TF-IDF" pada halaman utama.

Untuk menggunakan model ini dalam script Python:

```python
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load model dan vectorizer
model = joblib.load('models/best_model_svm_tfidf.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Fungsi preprocessing
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Prediksi sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    features = vectorizer.transform([preprocessed_text])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    # Map prediction to sentiment label
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_map[prediction]

    return sentiment, proba
```

## Performa Model

Model ini memiliki performa:

- Accuracy: ~85-90% (tergantung pada dataset)
- F1-score: ~0.85

## Informasi Tambahan

- Model ini dilatih menggunakan scikit-learn versi 1.0+
- Ukuran file model sekitar 10-20MB tergantung pada dataset pelatihan
- Model sudah dioptimasi untuk sentimen dalam bahasa Indonesia
