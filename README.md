# YouTube Comment Sentiment Analysis

Aplikasi analisis sentimen untuk komentar YouTube, menggunakan machine learning untuk menganalisis sentimen dan menampilkan visualisasi yang interaktif.

## Fitur

- 🔍 Ekstraksi komentar dari video YouTube melalui YouTube Data API
- 🧠 Analisis sentimen menggunakan model SVM dengan TF-IDF atau VADER
- 📊 Visualisasi hasil dengan chart dan word cloud
- 🔄 Pemrosesan data melalui server untuk keamanan dan performa lebih baik
- 🌐 Antarmuka modern dan responsif dengan Next.js

## Teknologi

### Frontend

- Next.js 14
- React
- TailwindCSS
- D3.js untuk visualisasi

### Backend

- FastAPI
- Scikit-learn
- NLTK
- Google API Client (YouTube Data API)

## Cara Menjalankan

### Prasyarat

- Node.js 18+
- Python 3.8+
- YouTube Data API Key

### Langkah-langkah

1. **Clone repositori**

   ```bash
   git clone https://github.com/yourusername/youtube-sentiment-analysis.git
   cd youtube-sentiment-analysis
   ```

2. **Setup Backend**

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   pip install -r requirements.txt

   # Tambahkan file .env dengan API key
   echo "YOUTUBE_API_KEY=your_youtube_api_key" > .env

   # Jalankan server
   uvicorn main:app --reload
   ```

3. **Setup Frontend**

   ```bash
   cd frontend
   npm install

   # Jalankan server development
   npm run dev
   ```

4. **Akses aplikasi**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/docs

## Cara Menggunakan

1. Masukkan URL video YouTube yang ingin dianalisis
2. Atur jumlah maksimum komentar yang ingin diambil
3. Pilih model analisis sentimen (SVM dengan TF-IDF atau VADER)
4. Klik "Analyze Comments" dan tunggu hasil analisis

## Model yang Digunakan

1. **SVM dengan TF-IDF**: Model Support Vector Machine yang dilatih dengan data komentar dan menggunakan TF-IDF untuk ekstraksi fitur text.
2. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: Model analisis sentimen berbasis leksikon yang khusus dikembangkan untuk teks media sosial.

## Struktur Proyek

```
youtube-sentiment-analysis/
├── backend/                # FastAPI backend
│   ├── models/             # Model machine learning
│   └── main.py             # Entry point aplikasi backend
├── frontend/               # Next.js frontend
│   ├── public/             # Aset statis
│   └── src/                # Kode sumber
│       ├── app/            # Halaman dan komponen
│       ├── actions/        # Server actions
│       └── services/       # Layanan API
└── README.md               # Dokumentasi
```

## Kontribusi

Kontribusi selalu diterima! Silakan buat issue atau pull request untuk perbaikan atau fitur baru.

## Lisensi

[MIT License](LICENSE)
