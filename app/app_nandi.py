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
    Esta herramienta utiliza **Redes Neuronales Convolucionales (CNN)** para identificar géneros musicales.
    **Elegí un método para empezar:**
""")

# --- SELECTOR DE ENTRADA ---
metodo = st.radio("Seleccioná cómo querés ingresar la música:", ["Enlace de Spotify", "Subir archivo MP3/WAV"])

link_input = None
archivo_subido = None

if metodo == "Enlace de Spotify":
    link_input = st.text_input("Pegá el enlace de Spotify acá:", placeholder="https://open.spotify.com/track/...")
else:
    archivo_subido = st.file_uploader("Seleccioná tu archivo de audio:", type=["mp3", "wav"])

if st.button("Analizar con Nandi"):
    ruta_temp = None
    cancion, artista = "Desconocido", "Desconocido"

    with st.spinner("Procesando audio..."):
        # LÓGICA 1: Si es por LINK
        if metodo == "Enlace de Spotify" and link_input:
            ruta_temp, cancion, artista = descargar_audio_nandi(link_input)
        
        # LÓGICA 2: Si es por ARCHIVO SUBIDO
        elif metodo == "Subir archivo MP3/WAV" and archivo_subido:
            if not os.path.exists('data/temp'):
                os.makedirs('data/temp', exist_ok=True)
            ruta_temp = "data/temp/subido_usuario.mp3"
            with open(ruta_temp, "wb") as f:
                f.write(archivo_subido.getbuffer())
            cancion = archivo_subido.name
            artista = "Archivo local"

        # --- PROCESAMIENTO CON LA IA ---
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
                
                if not resultados:
                    st.error("El archivo de audio es muy corto o no se pudo procesar.")
                else:
                    promedio = np.mean(resultados, axis=0)
                    idx_ganador = np.argmax(promedio)
                    genero_detectado = GENRES[idx_ganador].upper()
                    confianza = float(promedio[idx_ganador] * 100)

                    st.divider()
                    
                    if confianza < 50.0:
                        st.warning(f"⚠️ **Señal Ambigua:** El modelo detectó rasgos de **{genero_detectado}**, pero con confianza baja ({confianza:.2f}%).")
                    else:
                        st.success(f"✅ **Predicción Sólida:** He detectado **{genero_detectado}** con un {confianza:.2f}% de seguridad.")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Nombre:** {cancion}")
                        st.markdown(f"**Info:** {artista}")
                        st.progress(confianza / 100)
                    
                    with col2:
                        st.markdown("**Top 3 Probabilidades:**")
                        top3_idx = np.argsort(promedio)[-3:][::-1]
                        top3_data = pd.DataFrame({
                            'Género': [GENRES[i].upper() for i in top3_idx],
                            'Probabilidad': [float(promedio[i] * 100) for i in top3_idx]
                        })
                        st.bar_chart(top3_data, x='Género', y='Probabilidad')

                    # --- ESPECTROGRAMA ---
                    st.markdown("### 📊 Huella Sonora (Spectrogram)")
                    feat_plot = extraer_segmento_unificado(ruta_temp, 15) # Punto intermedio
                    if feat_plot is not None:
                        fig, ax = plt.subplots(figsize=(10, 3))
                        S_dB = feat_plot.reshape(128, 130)
                        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=22050, ax=ax, cmap='magma')
                        plt.colorbar(img, ax=ax, format='%+2.0f dB')
                        st.pyplot(fig)

                    # Guardar historial
                    registrar_prediccion("App Web", cancion, artista, genero_detectado.lower(), confianza, resultados)

            except Exception as e:
                st.error(f"Hubo un problema técnico: {e}")
        else:
            st.error("No se pudo obtener el audio. Si usaste un link, intentá subiendo el archivo directamente.")

st.divider()
st.caption("Nandi AI Project - Joel Gimenez")