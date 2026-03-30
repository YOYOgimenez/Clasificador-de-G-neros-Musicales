import streamlit as st
import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# --- CONFIGURACIÓN DE RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from predict_nandi import extraer_segmento_unificado, GENRES
from src.spotify_recommend import descargar_audio_nandi
from src.nandi_history import registrar_prediccion

MODEL_FILENAME = "v2_improved_model_best.h5"

st.set_page_config(page_title="Nandi AI Live", page_icon="🎷")

st.title("🎷 Nandi AI: Clasificación de Géneros")
st.markdown("Analizá música mediante **Redes Neuronales**. Si el link falla, subí el MP3.")

metodo = st.radio("Método:", ["Enlace de Spotify", "Subir archivo MP3/WAV"])

if st.button("Analizar con Nandi"):
    ruta_temp = None
    cancion, artista = "Desconocido", "Desconocido"

    with st.spinner("Procesando audio..."):
        if metodo == "Enlace de Spotify" and st.session_state.get('link_input'):
            ruta_temp, cancion, artista = descargar_audio_nandi(st.session_state.link_input)
        elif metodo == "Subir archivo MP3/WAV" and st.session_state.get('archivo_subido'):
            os.makedirs('data/temp', exist_ok=True)
            ruta_temp = os.path.join('data/temp', "subido_usuario.mp3")
            with open(ruta_temp, "wb") as f:
                f.write(st.session_state.archivo_subido.getbuffer())
            cancion = st.session_state.archivo_subido.name

        # --- CARGA DEL MODELO (PARCHE DE COMPATIBILIDAD) ---
        if ruta_temp and os.path.exists(ruta_temp):
            try:
                path_modelo = os.path.join(BASE_DIR, "models", MODEL_FILENAME)
                
                # PARCHE: Cargamos sin compilar para evitar el error de 'quantization_config'
                # que aparece en tu imagen f4f421.png
                model = tf.keras.models.load_model(path_modelo, compile=False)
                
                # Realizamos la predicción
                puntos = [15, 30, 45]
                resultados = []
                for p in puntos:
                    feat = extraer_segmento_unificado(ruta_temp, p)
                    if feat is not None:
                        preds = model.predict(feat, verbose=0)
                        resultados.append(preds[0])
                
                if resultados:
                    promedio = np.mean(resultados, axis=0)
                    idx_ganador = np.argmax(promedio)
                    genero = GENRES[idx_ganador].upper()
                    confianza = float(promedio[idx_ganador] * 100)

                    st.success(f"✅ Predicción: {genero} ({confianza:.2f}%)")
                    
                    # Gráfico de barras
                    top3_idx = np.argsort(promedio)[-3:][::-1]
                    top3_df = pd.DataFrame({
                        'Género': [GENRES[i].upper() for i in top3_idx],
                        'Probabilidad': [float(promedio[i] * 100) for i in top3_idx]
                    })
                    st.bar_chart(top3_df, x='Género', y='Probabilidad')
                    
                    # Espectrograma
                    fig, ax = plt.subplots(figsize=(10, 3))
                    feat_plot = extraer_segmento_unificado(ruta_temp, 20)
                    librosa.display.specshow(feat_plot.reshape(128, 130), x_axis='time', y_axis='mel', ax=ax)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error en el modelo: {e}")

# Captura de inputs fuera del botón para evitar reseteos
if metodo == "Enlace de Spotify":
    st.session_state.link_input = st.text_input("Link:")
else:
    st.session_state.archivo_subido = st.file_uploader("Archivo:", type=["mp3", "wav"])