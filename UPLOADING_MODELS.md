# Panduan Upload Model ke GitHub

Repositori ini menyertakan model machine learning untuk analisis sentimen. Berikut adalah panduan cara mengupload model agar aplikasi berfungsi dengan benar.

## Opsi 1: Upload Langsung ke GitHub (Untuk Model <50MB)

Jika ukuran model Anda kurang dari 50MB (batas ukuran file GitHub), Anda bisa mengupload file model langsung:

1. Pastikan file model berada di direktori `backend/models/`:

   - `best_model_svm_tfidf.pkl`
   - `tfidf_vectorizer.pkl`

2. Tambahkan file model ke repositori:

   ```bash
   git add backend/models/best_model_svm_tfidf.pkl
   git add backend/models/tfidf_vectorizer.pkl
   ```

3. Commit dan push perubahan:
   ```bash
   git commit -m "Add sentiment analysis model files"
   git push origin main
   ```

## Opsi 2: Menggunakan Git LFS (Untuk Model >50MB)

GitHub memiliki batas ukuran file 50MB untuk push langsung. Jika model Anda lebih besar, gunakan Git Large File Storage (LFS):

1. Instal Git LFS:

   ```bash
   # Windows
   git lfs install

   # macOS (dengan Homebrew)
   brew install git-lfs
   git lfs install

   # Linux (Debian/Ubuntu)
   sudo apt-get install git-lfs
   git lfs install
   ```

2. Inisialisasi Git LFS untuk file model:

   ```bash
   git lfs track "*.pkl"
   git add .gitattributes
   git commit -m "Configure Git LFS for model files"
   ```

3. Tambahkan model ke repositori:
   ```bash
   git add backend/models/best_model_svm_tfidf.pkl
   git add backend/models/tfidf_vectorizer.pkl
   git commit -m "Add ML model files via Git LFS"
   git push origin main
   ```

## Opsi 3: Host Model di Cloud Storage

Jika model sangat besar atau Anda tidak ingin menyimpannya di GitHub:

1. Upload model ke layanan cloud seperti Google Drive, AWS S3, atau Hugging Face:

   - Google Drive: Upload file dan atur sharing ke "Anyone with the link"
   - AWS S3: Upload file ke bucket publik
   - Hugging Face: Upload model ke Hub

2. Modifikasi `backend/main.py` untuk mengunduh model saat startup:

   ```python
   import os
   import requests

   # URLs untuk model
   MODEL_URL = "https://your-storage-link/best_model_svm_tfidf.pkl"
   VECTORIZER_URL = "https://your-storage-link/tfidf_vectorizer.pkl"

   # Path lokal
   MODEL_PATH = "models/best_model_svm_tfidf.pkl"
   VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

   # Download jika belum ada
   if not os.path.exists(MODEL_PATH):
       print(f"Downloading model from {MODEL_URL}...")
       os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
       r = requests.get(MODEL_URL)
       with open(MODEL_PATH, 'wb') as f:
           f.write(r.content)

   if not os.path.exists(VECTORIZER_PATH):
       print(f"Downloading vectorizer from {VECTORIZER_URL}...")
       os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
       r = requests.get(VECTORIZER_URL)
       with open(VECTORIZER_PATH, 'wb') as f:
           f.write(r.content)
   ```

## Opsi 4: Menyertakan Model dalam Release Assets

GitHub memungkinkan file ukuran besar sebagai bagian dari Release:

1. Buat release di GitHub dengan tag versi (misal: v1.0.0)
2. Upload file model sebagai "release assets"
3. Dalam aplikasi, tambahkan kode untuk mengunduh model dari URL release saat pertama kali dijalankan

## Perhatian Penting

- **Keamanan**: Pertimbangkan implikasi keamanan jika model berisi data sensitif
- **Pelatihan Ulang**: Sertakan script untuk melatih ulang model jika diperlukan
- **Dokumentasi**: Jelaskan proses pelatihan model di file `backend/models/README.md`
- **Dependencies**: Pastikan semua dependency untuk model tercantum di `requirements.txt`

## Referensi

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub Documentation: About large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
