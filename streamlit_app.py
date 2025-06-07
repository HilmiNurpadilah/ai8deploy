import streamlit as st
import numpy as np
import joblib
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image
import base64
import os
import subprocess

print("=== STREAMLIT APP STARTED ===")
# Mapping label Inggris ke Indonesia (edit sesuai kebutuhan)
label_mapping = {
    "Apple___Apple_scab": "Apel - Kudis Apel",
    "Apple___Black_rot": "Apel - Busuk Hitam",
    "Apple___Cedar_apple_rust": "Apel - Karat Apel Cedar",
    "Apple___healthy": "Apel - Sehat",
    "Blueberry___healthy": "Blueberry - Sehat",
    "Cherry___Powdery_mildew": "Ceri - Embun Tepung",
    "Cherry___healthy": "Ceri - Sehat",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Jagung - Bercak Daun Cercospora",
    "Corn___Common_rust": "Jagung - Karat Umum",
    "Corn___Northern_Leaf_Blight": "Jagung - Hawar Daun Utara",
    "Corn___healthy": "Jagung - Sehat",
    "Grape___Black_rot": "Anggur - Busuk Hitam",
    "Grape___Esca_(Black_Measles)": "Anggur - Esca (Campak Hitam)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Anggur - Hawar Daun (Bercak Daun Isariopsis)",
    "Grape___healthy": "Anggur - Sehat",
    "Orange___Haunglongbing_(Citrus_greening)": "Jeruk - Haunglongbing (Citrus Greening)",
    "Peach___Bacterial_spot": "Persik - Bercak Bakteri",
    "Peach___healthy": "Persik - Sehat",
    "Pepper,_bell___Bacterial_spot": "Paprika - Bercak Bakteri",
    "Pepper,_bell___healthy": "Paprika - Sehat",
    "Potato___Early_blight": "Kentang - Hawar Dini",
    "Potato___Late_blight": "Kentang - Hawar Lambat",
    "Potato___healthy": "Kentang - Sehat",
    "Raspberry___healthy": "Raspberry - Sehat",
    "Soybean___healthy": "Kedelai - Sehat",
    "Squash___Powdery_mildew": "Labu - Embun Tepung",
    "Strawberry___Leaf_scorch": "Stroberi - Daun Terbakar",
    "Strawberry___healthy": "Stroberi - Sehat",
    "Tomato_Bacterial_spot": "Tomat - Bercak Bakteri",
    "Tomato_Early_blight": "Tomat - Hawar Dini",
    "Tomato_Late_blight": "Tomat - Hawar Lambat",
    "Tomato_Leaf_Mold": "Tomat - Jamur Daun",
    "Tomato_Septoria_leaf_spot": "Tomat - Bercak Daun Septoria",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomat - Tungau Berbintik Dua",
    "Tomato__Target_Spot": "Tomat - Bercak Target",
    "Tomato__Tomato_mosaic_virus": "Tomat - Virus Mosaic",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomat - Virus Daun Kuning Keriting",
    "Tomato_healthy": "Tomat - Sehat",
}

# Download model otomatis jika belum ada
MODEL_PATH = 'klasifikasi_penyakit_daunnn/models/random_forest_model.pkl'
MODEL_DIR = os.path.dirname(MODEL_PATH)
MODEL_GDRIVE_ID = '1TiBzISDtQR4_vuyPr7hgSA0Iwh3wHnkH'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        import gdown
    except ImportError:
        subprocess.check_call(['pip', 'install', 'gdown'])
        import gdown
    url = f'https://drive.google.com/uc?id={MODEL_GDRIVE_ID}'
    print(f"Downloading model from {url} ...")
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = joblib.load(MODEL_PATH)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909763.png", width=80)
st.sidebar.title('Tentang Aplikasi')
# ... (lanjutkan dengan kode Streamlit kamu sesuai file asli)
