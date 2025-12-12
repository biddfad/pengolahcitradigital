import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 1. FUNGSI LOGIKA (BACKEND) ---

def rle_compression(data):
    if len(data) == 0: return []
    compressed = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i-1]:
            count += 1
        else:
            compressed.append((data[i-1], count))
            count = 1
    compressed.append((data[-1], count))
    return compressed

def rle_decompression(compressed):
    decompressed = []
    for value, count in compressed:
        decompressed.extend([value] * count)
    return decompressed

def proses_segmentasi(image_array):
    # Resize agar proses cepat
    img = cv2.resize(image_array, (200, 200))
    
    # Convert ke HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- PERBAIKAN DI SINI (SUPAYA BACKGROUND LEBIH BERSIH) ---
    # Kita naikkan batas bawah hijaunya sedikit biar bayangan kuning/coklat gak ikut
    lower_green = np.array([30, 40, 40]) 
    upper_green = np.array([90, 255, 255])
    
    # Masking
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Opsional: Membersihkan noise bintik-bintik kecil (Morphology)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Ekstraksi
    result = cv2.bitwise_and(img, img, mask=mask)
    return img, result

# --- 2. TAMPILAN APLIKASI (FRONTEND) ---

st.set_page_config(page_title="RLE Image Compressor", layout="wide")

st.title("🍊 Mini App: Kompresi Citra RLE")
st.write("Aplikasi sederhana untuk segmentasi objek dan kompresi Run-Length Encoding.")

# Sidebar Upload
st.sidebar.header("1. Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih gambar buah (Format JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    
    # STEP 1
    st.header("Step 1: Segmentasi & Preprocessing")
    col1, col2 = st.columns(2)
    img_asli, img_segmentasi = proses_segmentasi(img_cv)
    
    # Convert BGR ke RGB untuk ditampilkan
    img_asli_show = cv2.cvtColor(img_asli, cv2.COLOR_BGR2RGB)
    img_seg_show = cv2.cvtColor(img_segmentasi, cv2.COLOR_BGR2RGB)
    
    with col1:
        st.image(img_asli_show, caption="Gambar Asli", use_container_width=True)
    with col2:
        st.image(img_seg_show, caption="Hasil Segmentasi (Background Hitam)", use_container_width=True)

    # STEP 2
    st.header("Step 2: Kompresi RLE")
    if st.button("Lakukan Kompresi RLE"):
        with st.spinner('Sedang memproses...'):
            pixels = img_seg_show.flatten().tolist()
            compressed_data = rle_compression(pixels)
            
            ukuran_asli = len(pixels)
            ukuran_kompresi = len(compressed_data)
            rasio = (1 - (ukuran_kompresi / ukuran_asli)) * 100
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Total Piksel Asli", f"{ukuran_asli:,}")
            col_stat2.metric("Ukuran RLE (Tuple)", f"{ukuran_kompresi:,}")
            col_stat3.metric("Efisiensi Kompresi", f"{rasio:.2f}%")
            
            st.success("Kompresi Selesai!")
            
            st.text("Sampel Data Terkompresi (20 data pertama):")
            st.code(str(compressed_data[:20]), language='python')

            # STEP 3
            st.header("Step 3: Dekompresi (Pembuktian)")
            decompressed_pixels = rle_decompression(compressed_data)
            h, w, c = img_segmentasi.shape
            img_hasil = np.array(decompressed_pixels, dtype=np.uint8).reshape((h, w, c))
            st.image(img_hasil, caption="Hasil Dekompresi", width=300)

else:
    st.info("Silakan upload gambar di menu sebelah kiri untuk memulai.")