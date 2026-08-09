import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- DOWNLOAD NLTK ---
@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
download_nltk()

# --- KAMUS NORMALISASI ---
norm_dict = {
  # === PENYELAMATAN EFEK CLEANING (TARUH DI SINI) ===
    'maf': 'maaf',
    'sat': 'saat',
    'pra': 'parah',
    'eror':'error',
    'god':'good',
    'godjob':'goodjob',
    'manfat':'manfaat',
    'bermanfat':'manfaat',
    'mengunakan':'menggunakan',

    # kata umum / singkatan
    'bgt': 'banget', 'tmpt': 'tempat', 'tdk': 'tidak', 'jln': 'jalan',
    'bgus':'bagus', 'bgs':'bagus', 'mntap':'mantap', 'mntp':'mantap',
    'mantaf':'mantap','mantab':'mantap','bangus':'bagus','mantep':'mantap','mantaps':'mantap',
    'mantul':'mantap', 'mkasih':'terimakasih','gercep':'cepat','ancur':'hancur',
    # tambahan
    'ngebug':'bug','diblokir':'blokir','terblokir':'blokir','diblokir':'blokir','pemblokiran':'blokir',
    'rugikan':'rugi','nolak':'tolak','menolak':'tolak','ditolak':'tolak','menolong':'membantu','tertolong':'terbantu',
    'kren':'keren','ngga': 'tidak', 'ngaa': 'tidak', 'nga': 'tidak','jdi': 'jadi', 'jd': 'jadi',
    'siiip': 'sip', 'siip': 'sip','bosok': 'busuk', 'busok': 'busuk','mksh':'terimakasih','mksih':'terimakasih',
    # ========
    'merugikan':'rugi','dirugikan':'rugi',
    'jlk': 'jelek', 'butut':'jelek',
    'apk': 'aplikasi', 'app': 'aplikasi', 'apknya': 'aplikasi',
    'sy': 'saya', 'gw': 'saya', 'gue': 'saya', 'aku': 'saya',
    'ngk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'ora': 'tidak','engak':'tidak', 'enggak':'tidak','ngak':'tidak',
    'tp': 'tapi', 'udh': 'sudah', 'blm': 'belum',
    'jg': 'juga', 'krn': 'karena', 'trs': 'terus', 'lg': 'lagi',
    'yg': 'yang',
    'indriver': 'indrive',
    'aja': 'saja',
    'udah':'sudah', 'gk': 'tidak',
    'klo': 'kalau',
    'jd': 'jadi',
    'dpt': 'dapat',
    'utk': 'untuk',
    'dl': 'dulu',
    'dr': 'dari',
    'ok':'oke',
    'anyep':'sepi',
    "bugnya": "bug",
    "errornya": "error",
    "erornya": "eror",
    "sepinya": "sepi",
    "murahnya": "murah",


    # inDrive
    'indri': 'indrive', 'indrivee': 'indrive',

    # transport / aktor
    'drv': 'driver', 'penumpng': 'penumpang',
    'cust': 'pelanggan', 'pelanggannya': 'pelanggan', 'pelanggannyaa': 'pelanggan',
    'cs': 'customer service',

    # harga & pembayaran
    'hrga': 'harga', 'brp': 'berapa', 'mahalin': 'mahal', 'murmer': 'murah',
    'ongkir': 'ongkos', 'byr': 'bayar', 'bayr': 'bayar', 'cash': 'tunai', 'argo': 'harga',

    # perjalanan
    'jrak': 'jarak', 'deket': 'dekat', 'nunggu': 'menunggu', 'nuggu': 'menunggu',

    # keluhan / kondisi
    'btl': 'batal', 'lemot': 'lambat', 'telat': 'terlambat', 'ribet': 'rumit',
    'censel':'cancel', 'cansel':'cancel','elor':'error', 'eror': 'error', # Direvisi
}

# --- STOPWORDS HANDLING ---
stop_words = set(stopwords.words('indonesian'))
stop_words.update(['sih', 'nih', 'dong', 'lah', 'deh', 'kok', 'yah', 'pun', 'eh', 'ah', 'oh', 'hm', 'ya', 'iya','wkwk','dan','nya','untuk','kalau','kalo','sudah','karena','atau'])
kata_penting = {
        'tidak',
    'belum', 'kurang', 'jangan',
    'bukan', 'tapi',
    'sangat', 'terlalu', 'paling', 'lebih', 'cukup',
    'sebab', 'jika',
    'baik','malah',
    'justru',
    'padahal',
    'namun',
    'meski',
    'walaupun',
    'sempat',
    'masih',
    'sudah',
    'selalu',
    'sering',
    'jarang',
    'masalah',
    'menggunakan',
    # --- TAMBAHAN BARU HASIL BEDAH EXCEL ---
    'tepat', 'waktu', 'guna', 'adil', 'berubah', 'berat', 'bisa','banyak',
}
stop_words.difference_update(kata_penting)

# --- FUNGSI PREPROCESSING ---
def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1+', r'\1', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    normalized_words = [norm_dict.get(word, word) for word in words]
    text = " ".join(normalized_words)
    
    text = text.replace('very good', 'sangat bagus')
    text = text.replace('ojek online', 'ojol')
    text = text.replace('top up', 'isi saldo')
    text = text.replace('the best', 'terbaik')
    text = text.replace('gasopan', 'tidak sopan')
    text = text.replace('gabaik', 'tidak baik')
    text = text.replace('gajelas', 'tidak jelas')
    text = text.replace('gaada', 'tidak ada')
    text = text.replace('gabener', 'tidak benar')
    text = text.replace('gabisa', 'tidak bisa')


    text = text.replace('cukup bagus', 'bagus')
    text = text.replace('cukup baik', 'baik')
    text = text.replace('cukup membantu', 'membantu')

    text = text.replace('cukup mahal', 'mahal')
    text = text.replace('cukup lama', 'lambat')
    text = text.replace('cukup ribet', 'rumit')
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

# --- FUNGSI SAKTI N-GRAM ---
@st.cache_data
def get_top_ngram(corpus, n=None, ngram_range=(1, 1)):
    vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:n], columns=['Term', 'Frekuensi'])

# --- LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_bilstm_model():
    model = load_model('model_bilstm.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer
