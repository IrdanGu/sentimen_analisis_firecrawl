"""
model.py — Training & Inference Model Sentiment Analysis
Approach 1: TF-IDF + Logistic Regression (cepat, ringan)
Approach 2: HuggingFace IndoBERT (lebih akurat untuk bahasa Indonesia)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline


# =============================================================
# APPROACH 1: CLASSIC ML (TF-IDF + Logistic Regression)
# Cocok untuk: belajar konsep dasar, dataset kecil, resource terbatas
# =============================================================

class ClassicSentimentModel:
    def __init__(self):
        # Pipeline = TF-IDF vectorizer + Logistic Regression
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,    # Ambil 10k kata paling penting
                ngram_range=(1, 2),    # Unigram + Bigram
                min_df=2,              # Minimal muncul di 2 dokumen
                sublinear_tf=True      # Log scaling untuk TF
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ))
        ])
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}
        self.label_map_inv = {v: k for k, v in self.label_map.items()}

    def train(self, df: pd.DataFrame, text_col="text_clean", label_col="sentiment"):
        """
        Latih model dari DataFrame.
        """
        print("🚀 Memulai training Classic ML Model...")

        X = df[text_col].values
        y = df[label_col].map(self.label_map).values

        # Split data: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"   Training samples : {len(X_train)}")
        print(f"   Testing samples  : {len(X_test)}")

        # Training
        self.pipeline.fit(X_train, y_train)

        # Evaluasi
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n✅ Training selesai!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\n📊 Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=["negative", "neutral", "positive"]
        ))

        # Simpan hasil evaluasi
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

        return accuracy

    def predict(self, texts: list) -> list:
        """
        Prediksi sentiment dari list teks.
        Return: list of dicts dengan label dan confidence score.
        """
        if isinstance(texts, str):
            texts = [texts]

        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)

        results = []
        for pred, proba in zip(predictions, probabilities):
            label = self.label_map_inv[pred]
            confidence = proba[pred]
            results.append({
                "sentiment": label,
                "confidence": round(float(confidence), 4),
                "scores": {
                    "negative": round(float(proba[0]), 4),
                    "neutral": round(float(proba[1]), 4),
                    "positive": round(float(proba[2]), 4),
                }
            })
        return results

    def save(self, path="models/classic_model.pkl"):
        """Simpan model ke file."""
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"✅ Model tersimpan di {path}")

    def load(self, path="models/classic_model.pkl"):
        """Load model dari file."""
        self.pipeline = joblib.load(path)
        print(f"✅ Model berhasil di-load dari {path}")


# =============================================================
# APPROACH 2: HUGGINGFACE TRANSFORMER (IndoBERT)
# Cocok untuk: akurasi lebih tinggi, teks bahasa Indonesia
# Model: indobenchmark/indobert-base-p1
# =============================================================

class TransformerSentimentModel:
    def __init__(self, model_name="indobenchmark/indobert-base-p1"):
        """
        Gunakan pre-trained IndoBERT untuk bahasa Indonesia.
        Alternatif: "nlptown/bert-base-multilingual-uncased-sentiment"
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_pretrained(self):
        """
        Load model pre-trained dari HuggingFace Hub.
        (Download otomatis, butuh internet saat pertama kali)
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        print(f"📥 Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3  # negative, neutral, positive
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ Model loaded! Menggunakan: {self.device.upper()}")

    def predict(self, texts: list) -> list:
        """
        Prediksi sentiment menggunakan transformer model.
        """
        import torch

        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()
        results = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)

                outputs = self.model(**inputs)
                proba = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                pred = proba.argmax()
                labels = ["negative", "neutral", "positive"]

                results.append({
                    "sentiment": labels[pred],
                    "confidence": round(float(proba[pred]), 4),
                    "scores": {
                        "negative": round(float(proba[0]), 4),
                        "neutral": round(float(proba[1]), 4),
                        "positive": round(float(proba[2]), 4),
                    }
                })
        return results

    def save(self, path="models/transformer_model"):
        """Simpan fine-tuned model."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✅ Transformer model tersimpan di {path}")


# =============================================================
# CONTOH PENGGUNAAN
# =============================================================
if __name__ == "__main__":
    # --- Classic ML ---
    df = pd.read_csv("data/processed/clean_reviews.csv")

    model = ClassicSentimentModel()
    accuracy = model.train(df)
    model.save()

    # Test prediksi
    test_reviews = [
        "Produk sangat bagus, kualitas memuaskan dan pengiriman cepat!",
        "Kecewa banget, barang tidak sesuai deskripsi dan rusak.",
        "Lumayan, tidak terlalu bagus tapi juga tidak buruk.",
    ]

    predictions = model.predict(test_reviews)
    for review, pred in zip(test_reviews, predictions):
        print(f"\nReview : {review[:50]}...")
        print(f"Hasil  : {pred['sentiment']} ({pred['confidence']*100:.1f}% confidence)")
