import os
import sys
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

try:
    from nandi_utils import preparar_audio_nandi
except ImportError:
    from src.nandi_utils import preparar_audio_nandi

from src.spotify_recommend import descargar_audio_nandi, obtener_playlist_por_genero
from src.nandi_history import registrar_prediccion

load_dotenv()

# --- CONFIGURACIÓN V4 ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "nandi_v4_final_best.h5")

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Umbral mínimo de confianza para considerar la predicción válida
CONFIANZA_MINIMA = 30.0

# Modelo cargado una sola vez en memoria (no recarga en cada predicción)
_model_cache = None


def cargar_modelo():
    """Carga el modelo V4 una sola vez y lo cachea en memoria."""
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"❌ No se encuentra el modelo en {MODEL_PATH}\n"
                f"   Asegurate de haber corrido train_model.py primero."
            )
        print("🧠 Cargando modelo Nandi V4...")
        _model_cache = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modelo cargado.")
    return _model_cache


def calcular_offsets(ruta_audio):
    """
    Calcula offsets adaptativos según la duración REAL del audio cargado.
    Usa librosa.load para medir la duración efectiva del archivo descargado,
    evitando el bug donde get_duration lee metadatos de la canción completa
    aunque el archivo sea solo una preview de 30s.
    """
    import librosa
    try:
        # Cargamos solo 1 segundo para obtener sr, luego medimos con soundfile
        import soundfile as sf
        info = sf.info(ruta_audio)
        duration = info.duration
    except Exception:
        try:
            # Fallback: cargar el archivo completo y medir
            y, sr = librosa.load(ruta_audio, sr=22050)
            duration = len(y) / sr
        except Exception:
            duration = 28.0   # fallback conservador para previews de Spotify

    # Limitamos a 28s por seguridad (preview Spotify = 30s, ventana CNN = 3s)
    duration = min(duration, 28.0)

    margen = 3.0
    if duration <= margen * 2:
        return [0.0]

    puntos = [
        margen,
        (duration / 2) - margen / 2,
        duration - margen * 2,
    ]
    return [max(0.0, round(p, 1)) for p in puntos]


def ejecutar_nandi_v4(link_musica):
    print("\n" + "=" * 50)
    print("🚀 --- NANDI AI V4: ANÁLISIS DE PRECISIÓN ---")
    print("=" * 50)

    # 1. Descarga del audio
    ruta, cancion, artista = descargar_audio_nandi(link_musica)

    if not ruta or not os.path.exists(ruta):
        print("❌ Error: No se pudo obtener el audio.")
        return

    try:
        # 2. Carga del modelo (usa caché si ya estaba cargado)
        model = cargar_modelo()
        print(f"🎵 Analizando: {cancion} — {artista}")

        # 3. Offsets adaptativos según duración real del audio
        offsets = calcular_offsets(ruta)
        print(f"🔍 Analizando {len(offsets)} fragmentos en offsets: {offsets}s")

        predicciones = []
        for offset in offsets:
            feat = preparar_audio_nandi(ruta, offset=offset)
            if feat is not None:
                preds = model.predict(feat, verbose=0)

                # Validación: los scores deben sumar ~1 (softmax sano)
                if abs(preds[0].sum() - 1.0) < 0.1:
                    predicciones.append(preds[0])
                else:
                    print(f"⚠️  Scores inválidos en offset {offset}s, descartando fragmento.")

        if not predicciones:
            print("❌ Error: No se pudieron procesar fragmentos válidos.")
            return

        # 4. Promedio de scores y resultado final
        promedio_scores = np.mean(predicciones, axis=0)
        idx_ganador     = int(np.argmax(promedio_scores))
        genero_detectado = GENRES[idx_ganador]
        confianza        = float(promedio_scores[idx_ganador] * 100)

        # Top 3 géneros para contexto
        top3_idx = np.argsort(promedio_scores)[::-1][:3]
        print(f"\n{'='*50}")
        print(f"✨ RESULTADO: {genero_detectado.upper()}  ({confianza:.1f}% confianza)")
        print(f"\n📊 Top 3:")
        for i in top3_idx:
            bar = "█" * int(promedio_scores[i] * 20)
            print(f"   {GENRES[i]:12s} {bar} {promedio_scores[i]*100:.1f}%")
        print(f"{'='*50}")

        # Advertencia si la confianza es baja
        if confianza < CONFIANZA_MINIMA:
            print(f"⚠️  Confianza baja ({confianza:.1f}%). El género puede no ser preciso.")

        # 5. Registro en historial
        registrar_prediccion(
            fuente="Spotify/YT",
            cancion=cancion,
            artista=artista,
            genero=genero_detectado,
            confianza=confianza,
            scores_individuales=predicciones
        )

        # 6. Recomendación de playlist
        sugerencia = obtener_playlist_por_genero(genero_detectado)
        if sugerencia:
            print(f"\n🎧 Playlist sugerida: {sugerencia['nombre']}")
            print(f"🔗 {sugerencia['url']}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"❌ Error crítico: {e}")
    finally:
        # Limpieza del audio temporal descargado
        if ruta and os.path.exists(ruta):
            try:
                os.remove(ruta)
            except Exception:
                pass


if __name__ == "__main__":
    url_input = input("\n🔗 Pegá el link de la canción (Spotify o YT): ").strip()
    if url_input:
        ejecutar_nandi_v4(url_input)
    else:
        print("❌ No ingresaste ningún link.")