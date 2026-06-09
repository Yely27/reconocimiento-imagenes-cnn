# ============================================================
# app.py
# Aplicación web operativa con Streamlit en la nube
# ============================================================

# === PARCHE DE COMPATIBILIDAD DE RECONOCIMIENTO DE IMÁGENES ===
import keras
from keras.src.layers.core import dense

# Guardamos la función de inicialización original de la capa Dense
original_dense_init = dense.Dense.__init__

# Creamos una función modificada que elimina 'quantization_config' si existe
def patched_dense_init(self, *args, **kwargs):
    if 'quantization_config' in kwargs:
        kwargs.pop('quantization_config') # Lo elimina silenciosamente para evitar el error
    original_dense_init(self, *args, **kwargs)

# Reemplazamos la función original con nuestro parche
dense.Dense.__init__ = patched_dense_init
# ==============================================================

import os
import gdown
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

st.set_page_config(
    page_title="Reconocimiento Inteligente de Imágenes",
    layout="centered"
)

st.title("Sistema Inteligente de Reconocimiento de Imágenes")
st.write("Modelo basado en CNN, Transfer Learning, Fine-Tuning y EfficientNetB0.")

MODEL_PATH = "modelo_final_eurosat.keras"
IMG_SIZE = 224

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

@st.cache_resource
def cargar_modelo():
    # === DESCARGA AUTOMÁTICA DESDE GOOGLE DRIVE ===
    # Si el modelo no está en el servidor de la nube, lo descarga automáticamente
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando el modelo de IA desde Google Drive... Esto solo toma un momento la primera vez."):
            # ⚠️ REEMPLAZA ESTO: Pon aquí adentro el ID de tu archivo compartido de Google Drive
            id_drive = "1bs48rivfymyaRmm65Ux4kW7Np9J-WRK3" 
            
            url_descarga = f'https://drive.google.com/uc?id={id_drive}'
            gdown.download(url_descarga, MODEL_PATH, quiet=False)
    # ==============================================
    
    return tf.keras.models.load_model(MODEL_PATH)

modelo = cargar_modelo()

archivo = st.file_uploader(
    "Suba una imagen JPG, JPEG o PNG para realizar el reconocimiento:",
    type=["jpg", "jpeg", "png"]
)

if archivo is not None:
    img = Image.open(archivo).convert("RGB")
    st.image(img, caption="Imagen cargada", use_container_width=True)
    
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    inicio = time.time()
    prediccion = modelo.predict(img_array)
    fin = time.time()
    
    clase = class_names[np.argmax(prediccion)]
    probabilidad = np.max(prediccion) * 100
    tiempo = fin - inicio
    
    st.subheader("Resultado del reconocimiento")
    st.write(f"Clase predicha: **{clase}**")
    st.write(f"Probabilidad: **{probabilidad:.2f}%**")
    st.write(f"Tiempo de inferencia: **{tiempo:.4f} segundos**")
    st.write("Modelo utilizado: **EfficientNetB0 con Fine-Tuning**")
    
    probabilidades = {
        class_names[i]: float(prediccion[0][i])
        for i in range(len(class_names))
    }
    st.bar_chart(probabilidades)
