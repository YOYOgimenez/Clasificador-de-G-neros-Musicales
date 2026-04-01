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
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from nandi_utils import preparar_audio_nandi, SR, N_MELS, N_FRAMES
from src.spotify_recommend import descargar_audio_nandi, obtener_playlist_por_genero
from src.nandi_history import registrar_prediccion

# --- CONFIGURACIÓN ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "nandi_v4_final_best.h5")

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

GENRE_EMOJI = {
    'blues': '🎸', 'classical': '🎻', 'country': '🤠', 'disco': '🪩',
    'hiphop': '🎤', 'jazz': '🎷', 'metal': '🤘', 'pop': '🎵',
    'reggae': '🌿', 'rock': '🔥'
}

# --- CARGA DEL MODELO (una sola vez por sesión) ---
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ No se encuentra el modelo en {MODEL_PATH}. Corré train_model.py primero.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def calcular_offsets(ruta_audio):
    """
    Offsets adaptativos según duración REAL del archivo descargado.
    Usa soundfile para evitar el bug de get_duration que lee metadatos
    de la canción completa aunque el archivo sea una preview de 30s.
    """
    try:
        import soundfile as sf
        info = sf.info(ruta_audio)
        duration = info.duration
    except Exception:
        try:
            y, sr = librosa.load(ruta_audio, sr=22050)
            duration = len(y) / sr
        except Exception:
            duration = 28.0

    duration = min(duration, 28.0)  # techo de seguridad para previews Spotify

    margen = 3.0
    if duration <= margen * 2:
        return [0.0]
    return [
        round(margen, 1),
        round((duration / 2) - margen / 2, 1),
        round(duration - margen * 2, 1),
    ]


def generar_espectrograma_db(ruta_audio, offset=0.0):
    """
    Genera el espectrograma en dB real (sin normalizar) solo para visualización.
    Separado de preparar_audio_nandi para no mezclar el pipeline de predicción.
    """
    try:
        y, sr = librosa.load(ruta_audio, sr=SR, duration=3, offset=offset)
        S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                               hop_length=512, n_fft=2048)
        S_dB = librosa.power_to_db(S, ref=np.max)
        if S_dB.shape[1] >= N_FRAMES:
            return S_dB[:, :N_FRAMES]
        return np.pad(S_dB, ((0, 0), (0, N_FRAMES - S_dB.shape[1])), mode='constant',
                      constant_values=S_dB.min())
    except Exception:
        return None


# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="Nandi AI", page_icon="🎷", layout="wide")

st.title("🎷 Nandi AI — Clasificador de Géneros")
st.markdown("Analizá el ADN sonoro de tus canciones favoritas con Inteligencia Artificial.")

link_input = st.text_input(
    "🔗 Link de Spotify o YouTube:",
    placeholder="https://open.spotify.com/track/... o https://youtu.be/..."
)

if st.button("🚀 Analizar con Nandi", use_container_width=True):
    if not link_input.strip():
        st.warning("Pegá un link primero.")
        st.stop()

    with st.spinner("⏬ Descargando audio..."):
        ruta_temp, cancion, artista = descargar_audio_nandi(link_input.strip())

    if not ruta_temp or not os.path.exists(ruta_temp):
        st.error("❌ No pude bajar esa canción. ¿El link es correcto?")
        st.stop()

    try:
        model = cargar_modelo()

        with st.spinner("🧠 Nandi está analizando..."):
            offsets = calcular_offsets(ruta_temp)
            predicciones = []

            for offset in offsets:
                feat = preparar_audio_nandi(ruta_temp, offset=offset)
                if feat is not None:
                    preds = model.predict(feat, verbose=0)
                    if abs(preds[0].sum() - 1.0) < 0.1:
                        predicciones.append(preds[0])

        if not predicciones:
            st.error("❌ No se pudieron procesar fragmentos válidos del audio.")
            st.stop()

        # --- RESULTADO ---
        promedio      = np.mean(predicciones, axis=0)
        idx_ganador   = int(np.argmax(promedio))
        genero        = GENRES[idx_ganador]
        emoji         = GENRE_EMOJI.get(genero, "🎵")
        confianza     = float(promedio[idx_ganador] * 100)

        st.divider()

        # Cabecera del resultado
        if confianza >= 60.0:
            st.success(f"{emoji} **{genero.upper()}** — {confianza:.1f}% de confianza")
        else:
            st.warning(f"⚠️ Probablemente **{genero.upper()}** — señal ambigua ({confianza:.1f}%)")

        col_info, col_chart = st.columns([1, 2])

        with col_info:
            st.markdown(f"**🎵 Canción:** {cancion}")
            st.markdown(f"**🎤 Artista:** {artista}")
            st.markdown(f"**🔍 Fragmentos analizados:** {len(predicciones)}")
            st.progress(confianza / 100)

        with col_chart:
            st.markdown("**📊 Probabilidades por género:**")
            df = pd.DataFrame({
                'Género': [f"{GENRE_EMOJI.get(g,'')}{g.upper()}" for g in GENRES],
                'Probabilidad (%)': [round(float(p * 100), 2) for p in promedio]
            }).sort_values('Probabilidad (%)', ascending=False)
            st.bar_chart(df.set_index('Género'))

        # --- ESPECTROGRAMA EN dB ---
        st.markdown("### 🌊 Huella Sonora (Mel Spectrogram)")
        offset_mid = offsets[len(offsets) // 2]
        S_dB_plot  = generar_espectrograma_db(ruta_temp, offset=offset_mid)

        if S_dB_plot is not None:
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            img = librosa.display.specshow(
                S_dB_plot, x_axis='time', y_axis='mel',
                sr=SR, hop_length=512, ax=ax, cmap='magma'
            )
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title(f"Mel Spectrogram — {cancion}", color='white')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            st.pyplot(fig)
            plt.close(fig)

        # --- RECOMENDACIÓN DE SPOTIFY ---
        st.markdown("### 🎧 Recomendación de Playlist")
        with st.spinner("Buscando playlist en Spotify..."):
            sugerencia = obtener_playlist_por_genero(genero)

        if sugerencia:
            st.markdown(
                f"Te recomendamos **{sugerencia['nombre']}** "
                f"— [Abrir en Spotify 🔗]({sugerencia['url']})"
            )
        else:
            st.info("No encontramos una playlist en este momento.")

        # --- GUARDAR HISTORIAL ---
        registrar_prediccion(
            fuente="App Web",
            cancion=cancion,
            artista=artista,
            genero=genero,
            confianza=confianza,
            scores_individuales=predicciones
        )

    except Exception as e:
        st.error(f"❌ Error crítico: {e}")

    finally:
        if ruta_temp and os.path.exists(ruta_temp):
            try:
                os.remove(ruta_temp)
            except Exception:
                pass

st.divider()
st.caption("Nandi AI Project — Data Science Student")