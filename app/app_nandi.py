import streamlit as st
import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# Intentamos forzar que el sistema reconozca la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from predict_nandi import extraer_segmento_unificado, GENRES
    from src.spotify_recommend import descargar_audio_nandi
except ImportError:
    st.error("🚨 Error de estructura: No se encuentran 'predict_nandi.py' o 'src/'.")

st.set_page_config(page_title="Nandi AI Live", page_icon="🎷")

# --- FUNCIÓN: RADAR DE MODELOS (Busca en todo el proyecto) ---
@st.cache_resource
def buscar_y_cargar_modelo():
    # Buscamos de forma recursiva cualquier archivo .h5 que se parezca al nuestro
    archivos_h5 = list(Path(".").rglob("v2_improved_model_best.h5"))
    
    if not archivos_h5:
        # Si no lo encuentra, listamos qué es lo que SÍ ve el servidor para debug
        st.write("DEBUG - Archivos detectados en el servidor:")
        st.json([str(p) for p in Path(".").rglob("*") if p.is_file() and not str(p).startswith(".")][:20])
        return None

    ruta_encontrada = archivos_h5[0]
    try:
        # compile=False para evitar el error de versiones de Keras
        return tf.keras.models.load_model(str(ruta_encontrada), compile=False)
    except Exception as e:
        st.error(f"Error al abrir el modelo: {e}")
        return None

# --- INTERFAZ ---
st.title("🎷 Nandi AI: Clasificación de Géneros")
st.markdown("Si el link de Spotify falla (Error 403), subí tu MP3.")

# Cargamos el modelo al inicio
model = buscar_y_cargar_modelo()

metodo = st.radio("Entrada:", ["Subir archivo MP3/WAV", "Link de Spotify"])

ruta_temp = None
cancion = "Desconocido"

if metodo == "Subir archivo MP3/WAV":
    archivo_subido = st.file_uploader("Tu música:", type=["mp3", "wav"])
    if st.button("Analizar con Nandi") and archivo_subido:
        os.makedirs('data/temp', exist_ok=True)
        ruta_temp = os.path.join('data/temp', "temp_audio.mp3")
        with open(ruta_temp, "wb") as f:
            f.write(archivo_subido.getbuffer())
        cancion = archivo_subido.name
else:
    link = st.text_input("Link de Spotify:")
    if st.button("Analizar Link") and link:
        with st.spinner("Descargando..."):
            ruta_temp, cancion, _ = descargar_audio_nandi(link)

# --- PROCESAMIENTO ---
if ruta_temp and os.path.exists(ruta_temp):
    if model is None:
        st.error("🚨 Nandi AI no encontró su 'cerebro' (v2_improved_model_best.h5).")
    else:
        with st.spinner("La IA está procesando el audio..."):
            puntos = [10, 25, 45]
            resultados = []
            for p in puntos:
                feat = extraer_segmento_unificado(ruta_temp, p)
                if feat is not None:
                    preds = model.predict(feat, verbose=0)
                    resultados.append(preds[0])
            
            if resultados:
                promedio = np.mean(resultados, axis=0)
                idx = np.argmax(promedio)
                st.success(f"### 🎵 Género: {GENRES[idx].upper()} ({promedio[idx]*100:.2f}%)")
                
                # Gráfico de barras
                top3_idx = np.argsort(promedio)[-3:][::-1]
                st.bar_chart(pd.DataFrame({
                    'Género': [GENRES[i].upper() for i in top3_idx],
                    'Confianza': [float(promedio[i] * 100) for i in top3_idx]
                }), x='Género', y='Confianza')
                
                # Espectrograma
                fig, ax = plt.subplots(figsize=(10, 3))
                librosa.display.specshow(extraer_segmento_unificado(ruta_temp, 20).reshape(128, 130), ax=ax, cmap='magma')
                st.pyplot(fig)