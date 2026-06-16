import streamlit as st
import tensorflow as tf
import time
import numpy as np
import os
import gdown  # Librería óptima para descargar archivos grandes de Drive
from PIL import Image
from tensorflow.keras.preprocessing import image

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento Inteligente de Imágenes",
    layout="centered"
)

st.title("Sistema Inteligente de Reconocimiento de Imágenes")
st.write("Modelo basado en CNN, Transfer Learning, Fine-Tuning y EfficientNetB0.")

MODEL_PATH = "modelo_final_eurosat.keras"
IMG_SIZE = 224

# Extraído directamente de tu enlace de Google Drive
DRIVE_FILE_ID = "1k16Pgixb3Y6qMkWWWQKuk5Ltj8Ygd6J3"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

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

# Modificamos el caché para evitar problemas de hilos con TensorFlow
@st.cache_resource
def cargar_modelo():
    # Comprobamos si el archivo real ya existe en el servidor o si está vacío (como el de 133 Bytes)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        with st.spinner("Descargando el cerebro del modelo desde Google Drive (Esto solo tardará un momento la primera vez)..."):
            try:
                # Descarga el archivo de forma segura
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Hubo un problema al descargar el modelo: {e}")
                return None
                
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

modelo = cargar_modelo()

if modelo is not None:
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
else:
    st.error("No se pudo cargar el modelo de Inteligencia Artificial. Verifique los logs.")
