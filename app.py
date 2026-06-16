import streamlit as st
import tensorflow as tf
import time
import numpy as np
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
    # Eliminamos el parche antiguo y cargamos de forma nativa
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

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
