import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_top_ngram

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Visualisasi Dataset", page_icon="📊", layout="wide")

# ==========================================
# 2. INJEKSI CSS KUSTOM (TEMA SERAGAM)
# ==========================================
st.markdown("""
<style>
    /* Tarik konten ke atas */
    .block-container { padding-top: 2rem !important; }
    
    /* Styling Header Section biar seragam */
    .section-header { display: flex; align-items: center; margin-bottom: 0.8rem; margin-top: 1rem; }
    .section-dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; }
    .section-header h3 { margin: 0; font-size: 1.25rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HEADER HALAMAN
# ==========================================
st.markdown(
    "<h1 style='font-size: 2.2rem; margin-top: -10px; margin-bottom: 5px;'>Visualisasi Data Ulasan</h1>", 
    unsafe_allow_html=True
)

try:
    df_viz = pd.read_csv('dataset_indrive_berlabel_revisi.csv')
    df_viz['Hasil_bersih'] = df_viz['Hasil_bersih'].fillna('')

    sentimen_count = df_viz['label'].value_counts()
    jml_positif = sentimen_count.get('Positif', 0)
    jml_netral = sentimen_count.get('Netral', 0)
    jml_negatif = sentimen_count.get('Negatif', 0)

    # ==========================================
    # 4. BAGIAN METRIK (TOTAL DATA)
    # ==========================================
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background-color: #0078D7;"></div>
        <h3>Total Data per Kelas</h3>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric(label="Positif 😍", value=f"{jml_positif:,}".replace(',', '.'))
    c2.metric(label="Netral 😐", value=f"{jml_netral:,}".replace(',', '.'))
    c3.metric(label="Negatif 😡", value=f"{jml_negatif:,}".replace(',', '.'))
    
    st.divider()

    # ==========================================
    # 5. BAGIAN PIE CHART & TABEL
    # ==========================================
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("**Distribusi Kelas Sentimen**")
        # Ukuran chart dikecilin dikit biar ga makan tempat (figsize=5,5)
        fig1, ax1 = plt.subplots(figsize=(5, 5)) 
        
        # Biar warnanya sesuai: Positif=Hijau, Netral=Biru, Negatif=Merah (otomatis nyesuain urutan index)
        colors_map = {'Positif': '#4ade80', 'Netral': '#60a5fa', 'Negatif': '#f87171'}
        pie_colors = [colors_map.get(label, '#999999') for label in sentimen_count.index]
        
        ax1.pie(sentimen_count, labels=sentimen_count.index, autopct='%1.1f%%', 
                colors=pie_colors, startangle=90, textprops={'fontsize': 10})
        ax1.axis('equal')
        st.pyplot(fig1)
        
    with col2:
        st.markdown("**Cuplikan Dataset Latih (10 Data Acak)**")
        st.info("💡 Tabel ini merupakan dataset awal hasil pelabelan Lexicon yang digunakan untuk melatih model Bi-LSTM.")
        
        df_tabel = df_viz[['content','Hasil_bersih', 'label']].sample(10)
        df_tabel.columns = ['Ulasan Asli (Raw)','Hasil Preprocessing', 'Label Sentimen']
        st.dataframe(df_tabel, use_container_width=True, hide_index=True)

    st.divider()
    
    # ==========================================
    # 6. BAGIAN UNIGRAM (1 KATA)
    # ==========================================
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background-color: #f59e0b;"></div>
        <h3>Top 10 Kata yang Paling Sering Muncul (Unigram)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_uni1, col_uni2 = st.columns(2)
    
    with col_uni1:
        st.markdown("<p style='font-weight:600; color:#ef4444;'>🔴 Sentimen Negatif</p>", unsafe_allow_html=True)
        data_neg = df_viz[df_viz['label'] == 'Negatif']['Hasil_bersih']
        df_uni_neg = get_top_ngram(data_neg, n=10, ngram_range=(1,1))
        
        # Figsize diatur biar bar chartnya rapi dan compact
        fig_neg, ax_neg = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Frekuensi', y='Term', data=df_uni_neg, palette='Reds_r', ax=ax_neg)
        ax_neg.set_ylabel('') # Ilangin tulisan 'Term' biar bersih
        st.pyplot(fig_neg)

    with col_uni2:
        st.markdown("<p style='font-weight:600; color:#22c55e;'>🟢 Sentimen Positif</p>", unsafe_allow_html=True)
        data_pos = df_viz[df_viz['label'] == 'Positif']['Hasil_bersih']
        df_uni_pos = get_top_ngram(data_pos, n=10, ngram_range=(1,1))
        
        fig_pos, ax_pos = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Frekuensi', y='Term', data=df_uni_pos, palette='Greens_r', ax=ax_pos)
        ax_pos.set_ylabel('')
        st.pyplot(fig_pos)

    st.divider()

    # ==========================================
    # 7. BAGIAN BIGRAM (2 KATA)
    # ==========================================
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background-color: #8b5cf6;"></div>
        <h3>Top 10 Frasa 2-Kata yang Sering Muncul (Bigram)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_bi1, col_bi2 = st.columns(2)

    with col_bi1:
        st.markdown("<p style='font-weight:600; color:#ef4444;'>🔴 Keluhan Terbanyak (Negatif)</p>", unsafe_allow_html=True)
        df_bi_neg = get_top_ngram(data_neg, n=10, ngram_range=(2,2))
        
        fig_bi_neg, ax_bi_neg = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Frekuensi', y='Term', data=df_bi_neg, palette='flare', ax=ax_bi_neg)
        ax_bi_neg.set_ylabel('')
        st.pyplot(fig_bi_neg)

    with col_bi2:
        st.markdown("<p style='font-weight:600; color:#22c55e;'>🟢 Pujian Terbanyak (Positif)</p>", unsafe_allow_html=True)
        df_bi_pos = get_top_ngram(data_pos, n=10, ngram_range=(2,2))
        
        fig_bi_pos, ax_bi_pos = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Frekuensi', y='Term', data=df_bi_pos, palette='viridis', ax=ax_bi_pos)
        ax_bi_pos.set_ylabel('')
        st.pyplot(fig_bi_pos)
        
except FileNotFoundError:
    st.error("File dataset 'dataset_indrive_berlabel_revisi.csv' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py.")