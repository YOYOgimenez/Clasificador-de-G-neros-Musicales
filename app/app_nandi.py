import streamlit as st
import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# --- RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from predict_nandi import extraer_segmento_unificado, GENRES
from src.spotify_recommend import descargar_audio_nandi

MODEL_FILENAME = "v2_improved_model_best.h5"

st.set_page_config(page_title="Nandi AI Live", page_icon="🎷")
st.title("🎷 Nandi AI: Music Classifier")

# --- LÓGICA DE CARGA SEGURA ---
@st.cache_resource
def cargar_modelo_nandi(ruta):
    # Esto soluciona el error de 'quantization_config' forzando la carga limpia
    return tf.keras.models.load_model(ruta, compile=False)

# --- INTERFAZ ---
metodo = st.sidebar.radio("Método de entrada:", ["Link de Spotify", "Subir MP3"])

ruta_temp = None
cancion, artista = "Desconocida", "Desconocido"

if metodo == "Link de Spotify":
    link = st.text_input("Enlace de Spotify:")
    if st.button("Analizar Link"):
        with st.spinner("Descargando... (Si falla, usá la opción de Subir MP3)"):
            ruta_temp, cancion, artista = descargar_audio_nandi(link)
else:
    archivo = st.file_uploader("Subí tu tema (MP3/WAV):", type=["mp3", "wav"])
    if st.button("Analizar Archivo"):
        if archivo:
            os.makedirs('data/temp', exist_ok=True)
            ruta_temp = os.path.join('data/temp', "temp_audio.mp3")
            with open(ruta_temp, "wb") as f:
                f.write(archivo.getbuffer())
            cancion = archivo.name

# --- PROCESAMIENTO ---
if ruta_temp and os.path.exists(ruta_temp):
    try:
        path_modelo = os.path.join(BASE_DIR, "models", MODEL_FILENAME)
        model = cargar_modelo_nandi(path_modelo)

        # La lógica de los 3 puntos que mencionaste (puntos: 10s, 25s, 45s)
        puntos = [10, 25, 45]
        resultados = []
        
        with st.spinner("La IA está escuchando..."):
            for p in puntos:
                feat = extraer_segmento_unificado(ruta_temp, p)
                if feat is not None:
                    preds = model.predict(feat, verbose=0)
                    resultados.append(preds[0])
        
        if resultados:
            promedio = np.mean(resultados, axis=0)
            idx = np.argmax(promedio)
            confianza = float(promedio[idx] * 100)
            
            st.success(f"### 🎵 Género: {GENRES[idx].upper()} ({confianza:.2f}%)")
            
            # Visualización
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top Probabilidades:**")
                top3_idx = np.argsort(promedio)[-3:][::-1]
                top3_df = pd.DataFrame({
                    'Género': [GENRES[i].upper() for i in top3_idx],
                    'Confianza': [float(promedio[i] * 100) for i in top3_idx]
                })
                st.bar_chart(top3_df, x='Género', y='Confianza')
            
            with col2:
                st.write("**Huella de Frecuencia:**")
                fig, ax = plt.subplots(figsize=(5, 4))
                feat_plot = extraer_segmento_unificado(ruta_temp, 25)
                librosa.display.specshow(feat_plot.reshape(128, 130), cmap='magma', ax=ax)
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error de compatibilidad: {e}")
        st.info("Tip: Si el error persiste, es un tema de versiones de Keras en el servidor.")