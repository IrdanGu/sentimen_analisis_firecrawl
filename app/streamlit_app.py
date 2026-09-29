"""
streamlit_app.py — Web App untuk Demo Sentiment Analysis
Deploy gratis di: https://streamlit.io/cloud
"""

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessor import TextPreprocessor
from src.model import ClassicSentimentModel

# =============================================================
# CONFIG
# =============================================================
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# LOAD MODEL (cached agar tidak reload terus)
# =============================================================
@st.cache_resource
def load_model():
    model = ClassicSentimentModel()
    model.load("models/classic_model.pkl")
    return model

@st.cache_resource
def load_preprocessor():
    return TextPreprocessor()


# =============================================================
# HELPER FUNCTIONS
# =============================================================
def get_emoji(sentiment: str) -> str:
    return {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(sentiment, "❓")

def get_color(sentiment: str) -> str:
    return {"positive": "#4CAF50", "neutral": "#FF9800", "negative": "#F44336"}.get(sentiment, "#gray")

def plot_confidence_gauge(confidence: float, sentiment: str):
    """Gauge chart untuk confidence score."""
    color = get_color(sentiment)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": f"Confidence Score", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#FFEBEE"},
                {"range": [50, 75], "color": "#FFF8E1"},
                {"range": [75, 100], "color": "#E8F5E9"},
            ],
        },
        number={"suffix": "%", "font": {"size": 24}}
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_score_bars(scores: dict):
    """Bar chart untuk semua class scores."""
    labels = list(scores.keys())
    values = [v * 100 for v in scores.values()]
    colors = ["#F44336", "#FF9800", "#4CAF50"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside"
    ))
    fig.update_layout(
        title="Distribusi Skor per Kelas",
        yaxis=dict(range=[0, 110], title="Score (%)"),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# =============================================================
# MAIN APP
# =============================================================
def main():
    # Header
    st.title("🔍 Sentiment Analysis — Review Produk")
    st.markdown("Analisis sentimen review produk menggunakan ML. Powered by **Firecrawl + Scikit-learn**")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        mode = st.radio(
            "Mode Analisis",
            ["Single Review", "Batch Review (CSV)", "Coba Contoh"],
            index=0
        )
        st.divider()
        st.markdown("**Tentang Model:**")
        st.info("Model: TF-IDF + Logistic Regression\nData: Review produk e-commerce\nLanguage: Indonesia + English")

    # Load resources
    try:
        model = load_model()
        preprocessor = load_preprocessor()
    except FileNotFoundError:
        st.error("❌ Model belum di-train! Jalankan `src/model.py` terlebih dahulu.")
        st.stop()

    # ======================
    # MODE 1: Single Review
    # ======================
    if mode == "Single Review":
        st.subheader("📝 Analisis Satu Review")

        text_input = st.text_area(
            "Masukkan teks review:",
            placeholder="Contoh: Produk sangat bagus, kualitas memuaskan dan pengiriman cepat!",
            height=120
        )

        if st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("⚠️ Mohon masukkan teks review terlebih dahulu.")
            else:
                with st.spinner("Menganalisis..."):
                    clean_text = preprocessor.clean_text(text_input)
                    result = model.predict([clean_text])[0]

                sentiment = result["sentiment"]
                confidence = result["confidence"]
                scores = result["scores"]

                # Hasil utama
                col1, col2, col3 = st.columns(3)
                with col1:
                    emoji = get_emoji(sentiment)
                    color = get_color(sentiment)
                    st.markdown(f"""
                        <div style='text-align:center; padding:20px; border-radius:10px; background:{color}22; border: 2px solid {color}'>
                            <div style='font-size:48px'>{emoji}</div>
                            <div style='font-size:24px; font-weight:bold; color:{color}'>{sentiment.upper()}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    fig_gauge = plot_confidence_gauge(confidence, sentiment)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col3:
                    fig_bars = plot_score_bars(scores)
                    st.plotly_chart(fig_bars, use_container_width=True)

    # ======================
    # MODE 2: Batch CSV
    # ======================
    elif mode == "Batch Review (CSV)":
        st.subheader("📊 Analisis Batch dari CSV")
        st.info("Upload CSV dengan kolom 'text' berisi review yang ingin dianalisis.")

        uploaded_file = st.file_uploader("Upload file CSV", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                st.error("❌ CSV harus memiliki kolom bernama 'text'")
            else:
                st.success(f"✅ {len(df)} reviews ditemukan")
                st.dataframe(df.head())

                if st.button("🚀 Analisis Semua Review", type="primary"):
                    with st.spinner(f"Menganalisis {len(df)} reviews..."):
                        df["text_clean"] = df["text"].apply(preprocessor.clean_text)
                        predictions = model.predict(df["text_clean"].tolist())

                        df["sentiment"] = [p["sentiment"] for p in predictions]
                        df["confidence"] = [p["confidence"] for p in predictions]

                    st.success("✅ Analisis selesai!")

                    # Summary
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment_counts = df["sentiment"].value_counts()
                        fig_pie = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Distribusi Sentiment",
                            color=sentiment_counts.index,
                            color_discrete_map={
                                "positive": "#4CAF50",
                                "neutral": "#FF9800",
                                "negative": "#F44336"
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        avg_confidence = df["confidence"].mean()
                        st.metric("Total Reviews", len(df))
                        st.metric("Rata-rata Confidence", f"{avg_confidence*100:.1f}%")
                        st.metric("Review Positif", f"{(df['sentiment']=='positive').sum()} ({(df['sentiment']=='positive').mean()*100:.1f}%)")
                        st.metric("Review Negatif", f"{(df['sentiment']=='negative').sum()} ({(df['sentiment']=='negative').mean()*100:.1f}%)")

                    st.dataframe(df[["text", "sentiment", "confidence"]], use_container_width=True)

                    # Download hasil
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Hasil CSV",
                        csv,
                        "sentiment_results.csv",
                        "text/csv",
                        use_container_width=True
                    )

    # ======================
    # MODE 3: Contoh
    # ======================
    elif mode == "Coba Contoh":
        st.subheader("🎯 Coba Contoh Review")

        examples = {
            "Review Positif 1": "Produk sangat bagus sekali! Kualitas memuaskan, packaging aman, dan pengiriman cepat. Sudah 3x beli dan tidak pernah kecewa. Highly recommended!",
            "Review Positif 2": "Barang original, sesuai deskripsi. Seller ramah dan fast response. Puas banget belanja di sini!",
            "Review Netral": "Produknya lumayan sih. Kualitas standar, harga sesuai. Pengiriman agak lambat tapi barang aman sampai.",
            "Review Negatif 1": "Kecewa banget! Barang datang rusak dan tidak sesuai gambar. Sudah lapor tapi seller tidak responsif. Tidak akan beli lagi.",
            "Review Negatif 2": "Zonk! Barang palsu, kualitas sangat buruk. Jangan beli di sini!",
        }

        selected = st.selectbox("Pilih contoh review:", list(examples.keys()))
        example_text = examples[selected]
        st.text_area("Teks review:", value=example_text, height=100, disabled=True)

        if st.button("🔍 Analisis Contoh Ini", type="primary"):
            clean_text = preprocessor.clean_text(example_text)
            result = model.predict([clean_text])[0]

            sentiment = result["sentiment"]
            confidence = result["confidence"]

            emoji = get_emoji(sentiment)
            color = get_color(sentiment)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div style='text-align:center; padding:30px; border-radius:10px; background:{color}22; border: 2px solid {color}'>
                        <div style='font-size:60px'>{emoji}</div>
                        <div style='font-size:28px; font-weight:bold; color:{color}'>{sentiment.upper()}</div>
                        <div style='font-size:16px; color:gray'>Confidence: {confidence*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                fig = plot_score_bars(result["scores"])
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
