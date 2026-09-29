# 🔥 Sentiment Analysis dari Review Produk
> Firecrawl + Scikit-learn + HuggingFace + Streamlit | Full ML Pipeline

---

> Stack: Firecrawl + Python + Scikit-learn/HuggingFace + Streamlit
> Level: Intermediate ML | Goal: Portfolio + Deploy

---

## 📁 Struktur Project

```
sentiment-review-analyzer/
│
├── 📂 data/
│   ├── raw/              # Data mentah dari Firecrawl
│   ├── processed/        # Data setelah cleaning
│   └── labeled/          # Data setelah labeling
│
├── 📂 notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_and_cleaning.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── 📂 src/
│   ├── scraper.py        # Firecrawl scraping logic
│   ├── preprocessor.py   # Text cleaning & preprocessing
│   ├── model.py          # Model training & inference
│   └── utils.py          # Helper functions
│
├── 📂 app/
│   └── streamlit_app.py  # Web app untuk demo
│
├── 📂 models/
│   └── sentiment_model.pkl  # Saved trained model
│
├── requirements.txt
├── README.md
└── .env
```

---

## 🗺️ ROADMAP

### Phase 1 — Data Collection (Minggu 1)
Pakai Firecrawl untuk ambil review dari website.

### Phase 2 — EDA & Preprocessing (Minggu 1-2)
Bersihkan data, analisis, siapkan untuk training.

### Phase 3 — Model Training (Minggu 2-3)
Coba 2 approach: klasik (TF-IDF + Logistic Regression) dan modern (HuggingFace).

### Phase 4 — Evaluation & Improvement (Minggu 3)
Evaluasi model, tuning hyperparameter.

### Phase 5 — Deploy (Minggu 4)
Buat web app pakai Streamlit, deploy ke Streamlit Cloud (gratis!).

---

## ⚙️ SETUP

```bash
# Clone / buat project folder
mkdir sentiment-review-analyzer
cd sentiment-review-analyzer

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
firecrawl-py
pandas
numpy
scikit-learn
transformers
torch
streamlit
python-dotenv
matplotlib
seaborn
wordcloud
joblib
tqdm
```

**.env file:**
```
FIRECRAWL_API_KEY=fc-your-api-key-here
```
