# Menggunakan base image Python Slim (ringan)
FROM python:3.9-slim

# Instal dependensi sistem yang dibutuhkan OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Hapus file cache agar image lebih kecil
    && rm -rf /var/lib/apt/lists/*

# Set working directory di server
WORKDIR /app

# Salin file requirements.txt dan install dependencies Python
# Ini menginstal streamlit, opencv-python, numpy, dll.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file Streamlit ke folder kerja /app
COPY . .

# Perintah untuk menjalankan Streamlit
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app1.py"]