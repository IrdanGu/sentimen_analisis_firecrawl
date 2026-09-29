"""
preprocessor.py — Bersihkan dan siapkan teks review untuk training
"""

import re
import pandas as pd
import numpy as np


class TextPreprocessor:
    def __init__(self, language="indonesian"):
        self.language = language

        # Stopwords bahasa Indonesia sederhana
        # (bisa extend sesuai kebutuhan)
        self.stopwords_id = {
            "yang", "dan", "di", "ini", "itu", "tidak", "dengan",
            "untuk", "ke", "dari", "ada", "saya", "sih", "ya",
            "juga", "lebih", "sudah", "bisa", "sangat", "tapi",
            "karena", "kalau", "atau", "jadi", "aja", "nih", "deh",
            "loh", "dong", "kak", "nya"
        }

    def clean_text(self, text: str) -> str:
        """
        Bersihkan teks dari noise: URL, emoji, karakter aneh, dll.
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Hapus URL
        text = re.sub(r"http\S+|www\S+", "", text)

        # Hapus mention & hashtag (kalau dari social media)
        text = re.sub(r"@\w+|#\w+", "", text)

        # Hapus emoji & karakter non-latin
        text = re.sub(r"[^\w\s]", " ", text)

        # Hapus angka (opsional — bisa di-comment kalau rating penting)
        text = re.sub(r"\d+", "", text)

        # Hapus spasi berlebih
        text = " ".join(text.split())

        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        """Hapus stopwords dari teks."""
        words = text.split()
        filtered = [w for w in words if w not in self.stopwords_id]
        return " ".join(filtered)

    def auto_label(self, rating) -> str:
        """
        Buat label sentiment otomatis berdasarkan rating.
        1-2 bintang = negatif, 3 = netral, 4-5 = positif.
        """
        try:
            rating = float(rating)
            if rating >= 4:
                return "positive"
            elif rating == 3:
                return "neutral"
            else:
                return "negative"
        except (ValueError, TypeError):
            return None  # Rating tidak ada / tidak valid

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline lengkap preprocessing untuk DataFrame review.
        Input: DataFrame dengan kolom 'text' dan opsional 'rating'
        Output: DataFrame bersih siap training
        """
        print("🔄 Memulai preprocessing...")
        df = df.copy()

        # 1. Hapus baris yang kosong
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip() != ""]
        print(f"   ✅ Setelah hapus baris kosong: {len(df)} reviews")

        # 2. Bersihkan teks
        df["text_clean"] = df["text"].apply(self.clean_text)

        # 3. Hapus teks yang terlalu pendek (kurang dari 5 kata)
        df = df[df["text_clean"].str.split().str.len() >= 5]
        print(f"   ✅ Setelah filter teks pendek: {len(df)} reviews")

        # 4. Hapus duplikat
        df = df.drop_duplicates(subset=["text_clean"])
        print(f"   ✅ Setelah hapus duplikat: {len(df)} reviews")

        # 5. Auto-labeling dari rating (jika ada)
        if "rating" in df.columns:
            df["sentiment"] = df["rating"].apply(self.auto_label)
            df = df.dropna(subset=["sentiment"])
            print(f"   ✅ Setelah labeling: {len(df)} reviews")
            print(f"\n📊 Distribusi Sentiment:")
            print(df["sentiment"].value_counts())

        # 6. Simpan data bersih
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/clean_reviews.csv", index=False)
        print("\n✅ Data bersih tersimpan di data/processed/clean_reviews.csv")

        return df

    def get_text_stats(self, df: pd.DataFrame):
        """Tampilkan statistik teks untuk EDA."""
        df["word_count"] = df["text_clean"].str.split().str.len()
        df["char_count"] = df["text_clean"].str.len()

        print("📈 Statistik Teks:")
        print(f"   Rata-rata panjang review : {df['word_count'].mean():.1f} kata")
        print(f"   Review terpendek         : {df['word_count'].min()} kata")
        print(f"   Review terpanjang        : {df['word_count'].max()} kata")
        return df


import os

# =============================================================
# CONTOH PENGGUNAAN
# =============================================================
if __name__ == "__main__":
    preprocessor = TextPreprocessor(language="indonesian")

    # Simulasi data
    sample_data = pd.DataFrame({
        "text": [
            "Produk ini sangat bagus! Saya sangat puas dengan kualitasnya.",
            "Kecewa banget, barang rusak saat datang. Tidak recommended!",
            "Biasa aja sih, standar. Pengiriman lumayan cepat.",
            "Amazing product! Best purchase ever 🔥🔥",
            "",  # Ini akan dihapus
        ],
        "rating": [5, 1, 3, 4, None]
    })

    clean_df = preprocessor.preprocess_dataframe(sample_data)
    clean_df = preprocessor.get_text_stats(clean_df)
    print(clean_df[["text_clean", "sentiment", "word_count"]])
