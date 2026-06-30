import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Performa Model", page_icon="📈", layout="wide")

# ==========================================
# 2. INJEKSI CSS BIAR RAPI & NAIK KE ATAS
# ==========================================
st.markdown("""
<style>
    /* Tarik konten ke atas */
    .block-container { padding-top: 2rem !important; }
    
    /* Styling Header Section biar seragam sama app.py */
    .section-header { display: flex; align-items: center; margin-bottom: 1rem; margin-top: 1.5rem; }
    .section-dot { width: 12px; height: 12px; border-radius: 50%; background-color: #0078D7; margin-right: 10px; }
    .section-header h3 { margin: 0; font-size: 1.3rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HEADER & METRICS
# ==========================================
st.title("Evaluasi Performa Bi-LSTM")
st.write("Berdasarkan hasil pengujian di Google Colab, model mendapatkan evaluasi sebagai berikut:")

col1, col2, col3 = st.columns(3)
col1.metric("Akurasi Model", "93.77%")
col2.metric("Total Data Uji", "9.884 Baris")
col3.metric("Epoch Berhenti", "Early Stopping")

st.divider()

# ==========================================
# 4. BAGIAN GAMBAR GRAFIK (Lebar)
# ==========================================
st.markdown("""
<div class="section-header">
    <div class="section-dot"></div>
    <h3>Grafik Akurasi & Loss Pelatihan</h3>
</div>
""", unsafe_allow_html=True)

# Trik Kolom: [1, 4, 1] artinya kolom tengah ukurannya 4x lebih gede dari tepi
# Ini bikin gambarnya proporsional dan otomatis ke tengah
c1, c2, c3 = st.columns([1, 4, 1]) 
with c2:
    try:
        # Pake use_container_width buat ngilangin warning kuning!
        st.image("grafik_akurasi.png", use_container_width=True)
    except Exception:
        st.error("Gambar 'grafik_akurasi.png' tidak ditemukan.")

# ==========================================
# 5. BAGIAN GAMBAR CONFUSION MATRIX (Kotak)
# ==========================================
st.markdown("""
<div class="section-header">
    <div class="section-dot" style="background-color: #22c55e;"></div>
    <h3>Confusion Matrix</h3>
</div>
""", unsafe_allow_html=True)

# Trik Kolom: [1, 2, 1] karena confusion matrix biasanya bentuknya kotak, 
# jadi kolom tengahnya kita bikin lebih sempit biar gambarnya gak kegedean
c4, c5, c6 = st.columns([1, 2, 1]) 
with c5:
    try:
        st.image("confusion_matrix.png", use_container_width=True)
    except Exception:
        st.error("Gambar 'confusion_matrix.png' tidak ditemukan.")