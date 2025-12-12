import streamlit as st
import cv2
import numpy as np
import pandas as pd
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
    img = cv2.resize(image_array, (200, 200)) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40]) 
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(img, img, mask=mask)
    return img, result

# --- FUNGSI COPYRIGHT ---
def display_footer():
    # Menggunakan HTML/Markdown untuk memformat copyright
    current_year = 2024 # Anda bisa mengganti tahun ini secara manual atau menggunakan datetime.now().year
    footer_text = f"Mini App Kompresi RLE | Dibuat Oleh Abid Fadli J {current_year}"
    
    # Custom CSS untuk menempatkan footer di bagian bawah dan memisahkannya
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: #9E9E9E; font-size: small;'>{footer_text}</p>", 
                unsafe_allow_html=True)
# --- AKHIR FUNGSI COPYRIGHT ---


# --- 2. TAMPILAN APLIKASI (FRONTEND) ---

st.set_page_config(page_title="RLE Image Compressor", layout="centered")

# --- CUSTOM CSS INJECTION (dari revisi sebelumnya) ---
st.markdown("""
<style>
.stApp {
    background-color: #1E2024; /* Background utama */
    color: #FAFAFA;
}
h1, h2, h3, h4 {
    color: #FF5733 !important; 
}
.stAlert {
    background-color: #32353A !important;
    border-left: 5px solid #FF5733 !important;
    color: #FAFAFA !important;
}
</style>
""", unsafe_allow_html=True)

# Judul Utama & Subheader Dipusatkan
st.markdown("<h1 style='text-align: center; color: #FF5733;'> Mini App: Kompresi Citra RLE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FFA07A;'>Menggabungkan Segmentasi Citra & Run-Length Encoding</p>", unsafe_allow_html=True)
st.divider() 

# --- Sidebar Upload ---
st.sidebar.markdown(" 1. Upload Gambar")
uploaded_file = st.sidebar.file_uploader("Pilih gambar objek hijau (JPG/PNG)", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    # --- START PROCESSING ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    
    # STEP 1: SEGMENTASI
    st.markdown("---")
    st.subheader("1. Segmentasi Objek (Preprocessing RLE)")
    st.markdown("Memisahkan objek hijau dari *background* untuk menciptakan 'run' panjang (area hitam) agar RLE efisien.")
    
    col1, col2 = st.columns(2)
    img_asli, img_segmentasi = proses_segmentasi(img_cv)
    
    img_asli_show = cv2.cvtColor(img_asli, cv2.COLOR_BGR2RGB)
    img_seg_show = cv2.cvtColor(img_segmentasi, cv2.COLOR_BGR2RGB)
    
    with col1:
        st.image(img_asli_show, caption="Input (Resized 200x200 px)", use_container_width=True)
    with col2:
        st.image(img_seg_show, caption="Hasil Segmentasi", use_container_width=True)
        st.success("Objek berhasil diisolasi!")

    st.markdown("---")
    
    # STEP 2: KOMPRESI RLE
    st.subheader("2. Kompresi RLE & Analisis Efisiensi")
    
    if st.button("Lakukan Kompresi & Dekompresi"):
        with st.spinner('Sedang memproses RLE...'):
            pixels = img_seg_show.flatten().tolist()
            compressed_data = rle_compression(pixels)
            
            ukuran_asli = len(pixels)
            ukuran_kompresi = len(compressed_data)
            rasio = (1 - (ukuran_kompresi / ukuran_asli)) * 100
            
            # --- TAMPILAN STATISTIK DENGAN CHART ---
            st.markdown("##### Ringkasan Ukuran Data")
            
            df_stats = pd.DataFrame({
                'Tipe Data': ['Piksel Asli', 'Data RLE'],
                'Ukuran': [ukuran_asli, ukuran_kompresi]
            })
            
            st.bar_chart(df_stats, x='Tipe Data', y='Ukuran', color='#FF5733')
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Piksel Asli", f"{ukuran_asli:,} data")
            col_stat2.metric("Ukuran RLE (Tuple)", f"{ukuran_kompresi:,} data")
            col_stat3.metric("Efisiensi Kompresi", f"{rasio:.2f}%", delta_color="normal")
            
            st.success("Kompresi Selesai!")
            
            # --- TAMPILAN DATA SAMPEL ---
            with st.expander("Lihat Sampel Data Terkompresi"):
                st.markdown("Format: `(nilai_piksel, jumlah_berulang)`")
                st.code(str(compressed_data[:20]), language='python')

            # --- STEP 3: DEKOMPRESI ---
            st.markdown("---")
            st.subheader("3. Dekompresi (Pembuktian Lossless)")
            
            decompressed_pixels = rle_decompression(compressed_data)
            h, w, c = img_segmentasi.shape
            img_hasil = np.array(decompressed_pixels, dtype=np.uint8).reshape((h, w, c))
            
            st.image(img_hasil, caption="Hasil Dekompresi (Identik dengan Hasil Segmentasi)", width=300)
            st.info("Proses RLE bersifat *lossless*; hasil dekompresi sama persis dengan input segmentasi.")

else:
    # Pesan info awal yang dipusatkan
    st.info("Silakan **upload gambar** (disarankan buah dengan objek hijau) di menu sebelah kiri untuk memulai kompresi RLE.")


# --- PANGGIL FUNGSI COPYRIGHT DI BAGIAN PALING BAWAH ---
display_footer()