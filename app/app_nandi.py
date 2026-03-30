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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from predict_nandi import extraer_segmento_unificado, GENRES, MODEL_PATH
from src.spotify_recommend import descargar_audio_nandi
from src.nandi_history import registrar_prediccion

st.set_page_config(page_title="Nandi AI Live", page_icon="🎷")

st.title("🎷 Nandi AI: Clasificación de Géneros Musicales")
st.markdown("""
    Esta herramienta utiliza **Redes Neuronales Convolucionales (CNN)** para identificar géneros musicales 
    mediante el análisis de **Espectrogramas de Mel**. 
    La IA 've' las frecuencias y ritmos para determinar la categoría más probable.
""")

link_input = st.text_input("Enlace de Spotify:", placeholder="https://open.spotify.com/...")

if st.button("Analizar con Nandi"):
    if link_input:
        with st.spinner("Extrayendo características espectrales y procesando con el modelo..."):
            
            ruta_temp, cancion, artista = descargar_audio_nandi(link_input)
            
            if ruta_temp and os.path.exists(ruta_temp):
                try:
                    FULL_MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)
                    model = tf.keras.models.load_model(FULL_MODEL_PATH)
                    
                    puntos = [10, 25, 45]
                    resultados = []
                    
                    for p in puntos:
                        feat = extraer_segmento_unificado(ruta_temp, p)
                        if feat is not None:
                            preds = model.predict(feat, verbose=0)
                            resultados.append(preds[0])
                    
                    promedio = np.mean(resultados, axis=0)
                    idx_ganador = np.argmax(promedio)
                    genero_detectado = GENRES[idx_ganador].upper()
                    
                    # CORRECCIÓN FLOAT (Evita el error rojo)
                    confianza = float(promedio[idx_ganador] * 100)

                    st.divider()
                    
                    # --- LÓGICA DE INCERTIDUMBRE ---
                    if confianza < 50.0:
                        st.warning(f"⚠️ **Señal Ambigua:** El modelo detectó rasgos de **{genero_detectado}**, pero la confianza es baja ({confianza:.2f}%).")
                    else:
                        st.success(f"✅ **Predicción Sólida:** He detectado **{genero_detectado}** con un {confianza:.2f}% de seguridad.")

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Canción:** {cancion}")
                        st.markdown(f"**Artista:** {artista}")
                        st.progress(confianza / 100)
                    
                    with col2:
                        # Gráfico de barras del Top 3 para ver contra quién compite
                        st.markdown("**Top 3 Probabilidades:**")
                        top3_idx = np.argsort(promedio)[-3:][::-1]
                        top3_data = pd.DataFrame({
                            'Género': [GENRES[i].upper() for i in top3_idx],
                            'Probabilidad': [float(promedio[i] * 100) for i in top3_idx]
                        })
                        st.bar_chart(top3_data, x='Género', y='Probabilidad')

                    # --- ESPECTROGRAMA ---
                    st.markdown("###  Huella Sonora ( Spectrogram)")
                    feat_plot = extraer_segmento_unificado(ruta_temp, 25)
                    if feat_plot is not None:
                        fig, ax = plt.subplots(figsize=(10, 3))
                        S_dB = feat_plot.reshape(128, 130)
                        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=22050, ax=ax, cmap='magma')
                        plt.colorbar(img, ax=ax, format='%+2.0f dB')
                        st.pyplot(fig)

                    # Guardar historial
                    registrar_prediccion("App Web", cancion, artista, genero_detectado.lower(), confianza, resultados)

                except Exception as e:
                    st.error(f"Hubo un problema: {e}")
            else:
                st.error("No pude bajar esa canción. ¿El link es correcto?")

st.divider()
st.caption("Nandi AI Project - Joel Gimenez")