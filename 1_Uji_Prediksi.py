import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- IMPORT DARI utils.py ---
from utils import text_preprocessing, load_bilstm_model

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(page_title="Sentimen inDrive", page_icon="🚗", layout="wide")

# ==========================================
# 1.1 INJEKSI CSS KUSTOM BIAR DESAINNYA JALAN (SEIMBANG)
# ==========================================
st.markdown("""
<style>
    /* Tarik konten ke atas secukupnya */
    .block-container { padding-top: 2.5rem !important; }

    /* Styling Header Section */
    .section-header { display: flex; align-items: center; margin-bottom: 0.5rem; }
    .section-dot { width: 12px; height: 12px; border-radius: 50%; background-color: #0078D7; margin-right: 10px; }
    .section-header h3 { margin: 0; font-size: 1.3rem; font-weight: 600; }
    
    /* Styling Card Hasil - Padding dan font digedein dikit biar lega */
    .metric-card { padding: 1.2rem 1.5rem; border-radius: 12px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .metric-card.red { background: linear-gradient(135deg, #f87171, #ef4444); }
    .metric-card.blue { background: linear-gradient(135deg, #93c5fd, #3b82f6); }
    .metric-card.green { background: linear-gradient(135deg, #4ade80, #22c55e); }
    .metric-label { font-size: 0.95rem; opacity: 0.9; margin-bottom: 0.3rem; font-weight: 500; }
    .metric-value { font-size: 1.8rem; font-weight: bold; }
    .metric-sub { font-size: 0.85rem; background: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 4px; display: inline-block; margin-top: 4px; }
    
    /* Styling Box Teks Bersih - Ukuran normal */
    .result-box { background-color: rgba(128,128,128,0.1); padding: 1rem; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.95rem; border-left: 4px solid #0078D7; line-height: 1.4; }
    
    .stMarkdown { margin-bottom: -0.2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
model, tokenizer = load_bilstm_model()
MAX_LENGTH = 100 

# ==========================================
# 3. HEADER DASHBOARD
# ==========================================
# st.markdown(
#     "<p style='font-size: 18px; margin-top: 1px; margin-bottom: 15px;'>Uji Prediksi <b>Bidirectional LSTM (Bi-LSTM)</b>.</p>", 
#     unsafe_allow_html=True
# )

# ==========================================
# 4. TATA LETAK UTAMA (2 KOLOM)
# ==========================================
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div>
        <h3>Input Ulasan Uji Prediksi</h3>
    </div>
    """, unsafe_allow_html=True)

    # Inisialisasi session state buat contoh teks
    if 'example_text' not in st.session_state:
        st.session_state['example_text'] = ""

    # Tombol Contoh Cepat
    st.markdown("<p style='font-size:0.9rem; color:gray; margin-bottom:0.1rem; margin-top:0.5rem;'>Coba contoh cepat:</p>", unsafe_allow_html=True)
    eg_col1, eg_col2, eg_col3 = st.columns(3)
    with eg_col1:
        if st.button("😍 Positif", use_container_width=True):
            st.session_state['example_text'] = "Driver ramah dan cepat datang, tarifnya juga sangat murah!"
    with eg_col2:
        if st.button("😐 Netral", use_container_width=True):
            st.session_state['example_text'] = "Saya menggunakan aplikasi ini untuk pekerjaan sampingan"
    with eg_col3:
        if st.button("😡 Negatif", use_container_width=True):
            st.session_state['example_text'] = "Aplikasi sering error dan driver sering cancel pesanan!"

    # Input Area - Tinggi 150 (Pas, ga terlalu kecil)
    user_input = st.text_area(
        label="Masukkan teks:",
        height=150,
        placeholder="Contoh: Aplikasinya bagus banget, drivernya cepat datang dan ramah...",
        label_visibility="collapsed",
        value=st.session_state['example_text'] 
    )

    btn_predict = st.button("⚡ Prediksi Sekarang", type="primary", use_container_width=True)

with col_result:
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background-color: #22c55e;"></div>
        <h3>Hasil Analisis</h3>
    </div>
    <br>
    """, unsafe_allow_html=True)

    if btn_predict:
        if not user_input.strip():
            st.warning("⚠️ Masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Memproses prediksi..."):
                # 1. Preprocessing & Prediksi
                teks_bersih = text_preprocessing(user_input)
                seq = tokenizer.texts_to_sequences([teks_bersih])
                padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')
                probs = model.predict(padded)[0]
                kelas = int(np.argmax(probs))

                # 2. Konfigurasi Label Visual
                label_cfg = {
                    0: ("Negatif", "😡", "red",    "#EF4444"),
                    1: ("Netral",  "😐", "blue",   "#3b82f6"),
                    2: ("Positif", "😍", "green",  "#22c55e"),
                }
                nama, emoji, color, hex_color = label_cfg[kelas]
                conf = float(probs[kelas]) * 100

                # 3. Render Main Result Card
                st.markdown(f"""
                <div class="metric-card {color}">
                    <div class="metric-label">Prediksi Sentimen Model</div>
                    <div style="display:flex; align-items:center; gap:1.2rem;">
                        <span style="font-size:3rem; line-height:1;">{emoji}</span>
                        <div>
                            <div class="metric-value">{nama}</div>
                            <div class="metric-sub">Confidence: <strong>{conf:.1f}%</strong></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 4. Render Probability Bars
                st.markdown("<p style='font-size:0.95rem; font-weight:bold; margin-bottom:0.6rem;'>Detail Probabilitas:</p>", unsafe_allow_html=True)
                bar_data = [
                    ("Positif 😍", float(probs[2]), "#22c55e"),
                    ("Netral 😐",  float(probs[1]), "#3b82f6"),
                    ("Negatif 😡", float(probs[0]), "#EF4444"),
                ]
                
                # Jarak antar bar dikembalikan ke normal (gap:0.7rem) dan tinggi bar jadi 8px
                bars_html = '<div style="display:flex; flex-direction:column; gap:0.7rem; margin-bottom: 1.2rem;">'
                for lbl, val, clr in bar_data:
                    pct = val * 100
                    bars_html += f"""
                    <div>
                        <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:4px;">
                            <span>{lbl}</span>
                            <span style="font-family:monospace; color:{clr}; font-weight:bold;">{pct:.1f}%</span>
                        </div>
                        <div style="background:rgba(128,128,128,0.2); border-radius:99px; height:8px;">
                            <div style="width:{pct}%; background:{clr}; border-radius:99px; height:8px; transition:width 0.6s ease;"></div>
                        </div>
                    </div>"""
                bars_html += '</div>'
                st.markdown(bars_html, unsafe_allow_html=True)

                # 5. Render Preprocessing Result
                st.markdown(f"""
                <div>
                    <p style="font-size:0.85rem; color:gray; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;">
                        Teks Setelah Preprocessing
                    </p>
                    <div class="result-box">{teks_bersih if teks_bersih else '— (Kata dihilangkan karena Stopwords) —'}</div>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        # Placeholder saat belum diprediksi
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                    height:250px; background:rgba(128,128,128,0.05); border:2px dashed rgba(128,128,128,0.2);
                    border-radius:14px; gap:0.5rem; margin-top: 1rem;">
            <span style="font-size:3rem; opacity:0.4;">⚡</span>
            <p style="color:gray; font-size:0.95rem;">Hasil prediksi akan muncul di sini</p>
        </div>
        """, unsafe_allow_html=True)