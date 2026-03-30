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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Importamos tus funciones de lógica interna
try:
    from predict_nandi import extraer_segmento_unificado, GENRES
    from src.spotify_recommend import descargar_audio_nandi
except ImportError:
    st.error("No se pudieron cargar los módulos internos. Revisá que 'predict_nandi.py' esté en la raíz.")

MODEL_FILENAME = "v2_improved_model_best.h5"

st.set_page_config(page_title="Nandi AI Live", page_icon="🎷")

# --- FUNCIÓN: CARGA DEL MODELO A PRUEBA DE TODO ---
@st.cache_resource
def cargar_modelo_vikingo():
    # Buscamos en todas las rutas posibles que usa Streamlit Cloud
    posibles_paths = [
        os.path.join(BASE_DIR, "models", MODEL_FILENAME),
        os.path.join(CURRENT_DIR, "..", "models", MODEL_FILENAME),
        f"models/{MODEL_FILENAME}",
        MODEL_FILENAME
    ]
    
    for path in posibles_paths:
        if os.path.exists(path):
            try:
                # compile=False es CLAVE para evitar el error de 'quantization_config'
                return tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                continue
    return None

# --- INTERFAZ PRINCIPAL ---
st.title("🎷 Nandi AI: Clasificación de Géneros")
st.markdown("""
    IA entrenada con **Redes Neuronales (CNN)**. 
    *Nota: Si el link de Spotify falla (Error 403), por favor subí el archivo MP3.*
""")

metodo = st.radio("Seleccioná el método de entrada:", ["Link de Spotify", "Subir archivo MP3/WAV"])

# Inicializamos variables para que no den error si no se cargan
ruta_temp = None
cancion, artista = "Desconocido", "Desconocido"

if metodo == "Link de Spotify":
    link_input = st.text_input("Pegá el enlace acá:")
    if st.button("Analizar Link"):
        if link_input:
            with st.spinner("Intentando descargar..."):
                ruta_temp, cancion, artista = descargar_audio_nandi(link_input)
        else:
            st.warning("Por favor, ingresá un link.")

else:
    archivo_subido = st.file_uploader("Seleccioná tu archivo:", type=["mp3", "wav"])
    if st.button("Analizar Archivo"):
        if archivo_subido:
            with st.spinner("Cargando archivo..."):
                os.makedirs('data/temp', exist_ok=True)
                ruta_temp = os.path.join('data/temp', "temp_nandi.mp3")
                with open(ruta_temp, "wb") as f:
                    f.write(archivo_subido.getbuffer())
                cancion = archivo_subido.name
                artista = "Carga Manual"
        else:
            st.warning("Por favor, subí un archivo.")

# --- LÓGICA DE PROCESAMIENTO ---
if ruta_temp and os.path.exists(ruta_temp):
    try:
        model = cargar_modelo_vikingo()
        
        if model is None:
            st.error(f"🚨 No se encontró el archivo {MODEL_FILENAME}. Verificá tu carpeta 'models' en GitHub.")
        else:
            # Análisis en 3 puntos (10s, 25s, 45s) para mayor precisión
            puntos = [10, 25, 45]
            resultados = []
            
            with st.spinner("Nandi AI está analizando los patrones sonoros..."):
                for p in puntos:
                    feat = extraer_segmento_unificado(ruta_temp, p)
                    if feat is not None:
                        preds = model.predict(feat, verbose=0)
                        resultados.append(preds[0])
            
            if resultados:
                # Promediamos los resultados de los 3 puntos
                promedio = np.mean(resultados, axis=0)
                idx_ganador = np.argmax(promedio)
                genero_detectado = GENRES[idx_ganador].upper()
                confianza = float(promedio[idx_ganador] * 100)

                st.divider()
                st.success(f"### 🎵 Género Detectado: **{genero_detectado}**")
                st.info(f"Nivel de confianza: **{confianza:.2f}%**")

                # --- VISUALIZACIÓN ---
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top 3 Probabilidades:**")
                    top3_idx = np.argsort(promedio)[-3:][::-1]
                    top3_data = pd.DataFrame({
                        'Género': [GENRES[i].upper() for i in top3_idx],
                        'Probabilidad': [float(promedio[i] * 100) for i in top3_idx]
                    })
                    st.bar_chart(top3_data, x='Género', y='Probabilidad')
                
                with col2:
                    st.markdown("**Espectrograma de Mel:**")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    # Usamos el punto medio para el gráfico
                    feat_plot = extraer_segmento_unificado(ruta_temp, 25)
                    if feat_plot is not None:
                        S_dB = feat_plot.reshape(128, 130)
                        img = librosa.display.specshow(S_dB, cmap='magma', ax=ax)
                        st.pyplot(fig)

            else:
                st.error("El audio es muy corto para ser analizado en los 3 puntos.")

    except Exception as e:
        st.error(f"Hubo un problema técnico: {e}")
        st.info("Tip: Revisá que las versiones en requirements.txt sean correctas.")

st.divider()
st.caption("Nandi AI Project | Estudiante de Ciencia de Datos - UBA")