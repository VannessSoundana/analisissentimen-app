import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pickle
import os

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard Analisis Sentimen MyPertamina",
    page_icon="⛽",
    layout="wide"
)

# --- Judul Dashboard ---
st.title("⛽ Analisis Sentimen Ulasan Aplikasi MyPertamina")
st.markdown("Dashboard ini menampilkan hasil analisis sentimen dari ulasan pengguna di Google Play Store.")

# --- Memuat Model dan TF-IDF Vectorizer ---
# Tentukan path file untuk model dan TF-IDF vectorizer
model_path = r'D:\tes1\model\sentiment_model_oke.pkl'
tfidf_path = r'D:\tes1\model\tfidf_vectorizer_oke.pkl'

# Periksa apakah file path valid
if os.path.exists(model_path) and os.path.exists(tfidf_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(tfidf_path, 'rb') as tfidf_file:
        tfidf = pickle.load(tfidf_file)
    st.success("Model dan TF-IDF vectorizer berhasil dimuat!")
else:
    st.error(f"Path file tidak ditemukan: {model_path} atau {tfidf_path}")
    st.stop()  # Menghentikan eksekusi jika file tidak ditemukan

# Fungsi untuk prediksi sentimen menggunakan model yang sudah dilatih
def predict_sentiment(text):
    text_tfidf = tfidf.transform([text])  # Mengubah teks menjadi representasi numerik
    sentiment = model.predict(text_tfidf)  # Prediksi sentimen menggunakan model
    return sentiment[0]

# --- Fungsi Preprocessing Teks ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Upload File CSV ---
st.sidebar.header("Upload Data CSV")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV untuk analisis sentimen", type=["csv"])

if uploaded_file is not None:
    # Membaca file CSV yang diunggah
    data = pd.read_csv(uploaded_file)

    # Memastikan ada kolom yang dibutuhkan dalam data
    TEXT_COLUMN = 'clean_text'
    DATE_COLUMN = 'date'
    
    if TEXT_COLUMN not in data.columns:
        st.error(f"Kolom '{TEXT_COLUMN}' tidak ditemukan dalam data Anda. Harap sesuaikan nama kolom ulasan di kode.")
        st.stop()

    # --- Prediksi Sentimen Menggunakan Model ---
    if 'sentiment' not in data.columns:
        st.info("Kolom 'sentiment' tidak ditemukan, melakukan prediksi sentimen pada data ulasan...")
        try:
            # Melakukan preprocessing teks
            data['cleaned_review_for_pred'] = data[TEXT_COLUMN].astype(str).apply(preprocess_text)
            # Prediksi sentimen menggunakan model
            data['sentiment'] = data['cleaned_review_for_pred'].apply(predict_sentiment)
            st.success("Prediksi sentimen selesai!")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi sentimen. {e}")
            st.stop()
    else:
        st.info("Kolom 'sentiment' sudah ada dalam data Anda.")

    # --- Statistik Dasar Sentimen ---
    st.header("📊 Ringkasan Sentimen")
    sentiment_counts = data['sentiment'].value_counts()
    st.write(sentiment_counts)

    # Visualisasi Pie Chart
    st.subheader("Distribusi Sentimen")
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    colors = {'positif': 'lightgreen', 'negatif': 'lightcoral', 'netral': 'lightgray'}
    pie_colors = [colors.get(label, col) for label, col in zip(sentiment_counts.index, sns.color_palette("pastel", len(sentiment_counts)))]
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=pie_colors, textprops={'fontsize': 12})
    ax1.axis('equal')
    st.pyplot(fig1)

    # --- Tren Sentimen Seiring Waktu ---
    st.header("📈 Tren Sentimen Seiring Waktu")
    if DATE_COLUMN in data.columns:
        try:
            data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
            data_daily = data.groupby([data[DATE_COLUMN].dt.date, 'at']).size().unstack(fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            data_daily.plot(kind='line', ax=ax2, marker='o')
            ax2.set_title('Distribusi Sentimen Harian', fontsize=16)
            ax2.set_xlabel('Tanggal', fontsize=12)
            ax2.set_ylabel('Jumlah Ulasan', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(title='at')
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membuat grafik tren sentimen. {e}")
    else:
        st.info(f"Kolom '{DATE_COLUMN}' (tanggal) tidak ditemukan dalam data Anda.")

    # --- Ulasan Teratas Berdasarkan Sentimen ---
    st.header("📝 Ulasan Teratas")
    if 'sentiment' in data.columns:
        sentiment_options = ['positif', 'netral', 'negatif']
        selected_sentiment = st.selectbox("Pilih Sentimen untuk Dilihat:", options=sentiment_options)
        filtered_reviews = data[data['sentiment'] == selected_sentiment]
        if not filtered_reviews.empty:
            sample_reviews = filtered_reviews.sample(min(10, len(filtered_reviews)), random_state=42)
            for i, row in sample_reviews.iterrows():
                st.markdown(f"- **Ulasan:** *{row[TEXT_COLUMN]}*")
            st.markdown("---")
        else:
            st.info(f"Tidak ada ulasan dengan sentimen {selected_sentiment}.")
    else:
        st.warning("Kolom 'sentiment' tidak ditemukan atau kosong. Tidak dapat menampilkan ulasan teratas.")

    # --- Analisis Sentimen dari Input Manual ---
st.header("✍️ Coba Analisis Sentimen Manual")
st.markdown("Masukkan ulasan secara manual dan lihat prediksi sentimennya:")

user_input = st.text_area("Tulis komentar atau ulasan Anda di sini", "")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        preprocessed = preprocess_text(user_input)
        prediction = predict_sentiment(preprocessed)

        # --- Tampilan hasil prediksi dengan gaya menarik ---
        if prediction == "positif":
            st.markdown("### 🎉 Sentimen: **Positif**")
            st.success("Komentar ini mengandung sentimen **positif**. Pengguna tampaknya puas atau senang.")
        elif prediction == "negatif":
            st.markdown("### 😠 Sentimen: **Negatif**")
            st.error("Komentar ini mengandung sentimen **negatif**. Pengguna menunjukkan ketidakpuasan atau masalah.")
        elif prediction == "netral":
            st.markdown("### 😐 Sentimen: **Netral**")
            st.info("Komentar ini bersifat **netral**. Tidak terlalu positif maupun negatif.")
        else:
            st.warning(f"Sentimen tidak dikenali: {prediction}")

    st.markdown("---")
    st.info("Dashboard ini dibangun menggunakan Streamlit untuk analisis sentimen.")

