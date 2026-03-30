import streamlit as st
import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# --- CONFIGURACIÓN DE RUTAS DINÁMICAS ---
# Esto ayuda a que el servidor de Streamlit encuentre las carpetas 'models' y 'src'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) # Sube un nivel desde 'app/' a la raíz

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from predict_nandi import extraer_segmento_unificado, GENRES
from src.spotify_recommend import descargar_audio_nandi
from src.nandi_history import registrar_prediccion

# Nombre exacto del modelo que vimos en tu captura
MODEL_FILENAME = "v2_improved_model_best.h5"

st.set_page_config(page_title="Nandi AI Live", page_icon="🎷")

st.title("🎷 Nandi AI: Clasificación de Géneros Musicales")
st.markdown("""
    Esta herramienta utiliza **Redes Neuronales Convolucionales (CNN)**.
    Si el enlace de Spotify falla por restricciones de red, **subí el MP3 directamente**.
""")

# --- SELECTOR DE ENTRADA ---
metodo = st.radio("Seleccioná el método:", ["Enlace de Spotify", "Subir archivo MP3/WAV"])

link_input = None
archivo_subido = None

if metodo == "Enlace de Spotify":
    link_input = st.text_input("Pegá el enlace acá:", placeholder="https://open.spotify.com/track/...")
else:
    archivo_subido = st.file_uploader("Seleccioná tu audio:", type=["mp3", "wav"])

if st.button("Analizar con Nandi"):
    ruta_temp = None
    cancion, artista = "Desconocido", "Desconocido"

    with st.spinner("Procesando audio e iniciando IA..."):
        # 1. OBTENCIÓN DEL AUDIO
        if metodo == "Enlace de Spotify" and link_input:
            ruta_temp, cancion, artista = descargar_audio_nandi(link_input)
        
        elif metodo == "Subir archivo MP3/WAV" and archivo_subido:
            os.makedirs('data/temp', exist_ok=True)
            ruta_temp = os.path.join('data/temp', "subido_usuario.mp3")
            with open(ruta_temp, "wb") as f:
                f.write(archivo_subido.getbuffer())
            cancion = archivo_subido.name
            artista = "Carga Manual"

        # 2. CARGA DEL MODELO (RUTA BLINDADA)
        if ruta_temp and os.path.exists(ruta_temp):
            try:
                # Intentamos 3 rutas posibles para no fallar
                posibles_rutas = [
                    os.path.join(BASE_DIR, "models", MODEL_FILENAME),
                    os.path.join(CURRENT_DIR, "..", "models", MODEL_FILENAME),
                    f"models/{MODEL_FILENAME}"
                ]
                
                path_final_modelo = None
                for p in posibles_rutas:
                    if os.path.exists(p):
                        path_final_modelo = p
                        break
                
                if not path_final_modelo:
                    st.error(f"🚨 No se encuentra el archivo {MODEL_FILENAME} en la carpeta 'models'.")
                    st.stop()

                model = tf.keras.models.load_model(path_final_modelo)
                
                # 3. PREDICCIÓN
                puntos = [10, 25, 45]
                resultados = []
                for p in puntos:
                    feat = extraer_segmento_unificado(ruta_temp, p)
                    if feat is not None:
                        preds = model.predict(feat, verbose=0)
                        resultados.append(preds[0])
                
                if resultados:
                    promedio = np.mean(resultados, axis=0)
                    idx_ganador = np.argmax(promedio)
                    genero_detectado = GENRES[idx_ganador].upper()
                    confianza = float(promedio[idx_ganador] * 100)

                    st.divider()
                    if confianza < 50.0:
                        st.warning(f"⚠️ **Resultado Incierto:** {genero_detectado} ({confianza:.2f}%)")
                    else:
                        st.success(f"✅ **Predicción:** {genero_detectado} ({confianza:.2f}%)")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Tema:** {cancion}")
                        st.progress(confianza / 100)
                    
                    with col2:
                        top3_idx = np.argsort(promedio)[-3:][::-1]
                        top3_data = pd.DataFrame({
                            'Género': [GENRES[i].upper() for i in top3_idx],
                            'Probabilidad': [float(promedio[i] * 100) for i in top3_idx]
                        })
                        st.bar_chart(top3_data, x='Género', y='Probabilidad')

                    # --- ESPECTROGRAMA ---
                    st.markdown("### 📊 Análisis de Frecuencias (Mel Spectrogram)")
                    feat_plot = extraer_segmento_unificado(ruta_temp, 20)
                    if feat_plot is not None:
                        fig, ax = plt.subplots(figsize=(10, 3))
                        S_dB = feat_plot.reshape(128, 130)
                        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=22050, ax=ax, cmap='magma')
                        plt.colorbar(img, ax=ax, format='%+2.0f dB')
                        st.pyplot(fig)

                    registrar_prediccion("Web", cancion, artista, genero_detectado.lower(), confianza, resultados)
                else:
                    st.error("No se pudo procesar el contenido del audio.")

            except Exception as e:
                st.error(f"Error técnico en el modelo: {e}")
        else:
            st.error("No se pudo obtener el audio para analizar.")

st.divider()
st.caption("Nandi AI Project - Joel Gimenez | UBA Data Science student")