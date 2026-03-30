import os
import librosa
import numpy as np
import tensorflow as tf
from src.spotify_recommend import descargar_audio_nandi, obtener_playlist_por_genero
from src.nandi_history import registrar_prediccion  # Importado correctamente

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/v2_improved_model_best.h5"
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def extraer_segmento_unificado(ruta_audio, offset_seg):
    """ Función interna para procesar fragmentos específicos del audio bajado """
    try:
        y, sr = librosa.load(ruta_audio, sr=22050, duration=3, offset=offset_seg)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S)
        
        # Ajuste a 130 columnas
        if S_dB.shape[1] > 130:
            S_dB = S_dB[:, :130]
        else:
            S_dB = np.pad(S_dB, ((0, 0), (0, 130 - S_dB.shape[1])), mode='constant')
            
        return S_dB.reshape(1, 128, 130, 1)
    except Exception as e:
        return None

def ejecutar_nandi_completo(link_spotify):
    print(f"\n --- Nandi AI: Iniciando Predicción Inteligente ---")
    
    # FASE 1: Descargar el audio temporal
    ruta, cancion, artista = descargar_audio_nandi(link_spotify)
    
    if not ruta:
        print(" Error: No se pudo obtener el audio de Spotify.")
        return

    print(f" Analizando múltiples capas sonoras de: {cancion} - {artista}...")

    try:
        # FASE 2: Cargar el modelo
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # FASE 3: Análisis en 3 puntos (segundos 10, 25 y 45) - MANTENEMOS ESTO
        puntos = [10, 25, 45]
        resultados = []

        for p in puntos:
            feat = extraer_segmento_unificado(ruta, p)
            if feat is not None:
                preds = model.predict(feat, verbose=0)
                resultados.append(preds[0])

        if not resultados:
            print(" Error al procesar los fragmentos del audio.")
            return

        # FASE 4: Promedio y Decisión Final
        promedio = np.mean(resultados, axis=0)
        idx_ganador = np.argmax(promedio)
        genero_detectado = GENRES[idx_ganador]
        confianza = promedio[idx_ganador] * 100

        print(f"\n RESULTADO: Nandi detectó {genero_detectado.upper()} ({confianza:.2f}%)")

        # --- FASE NUEVA: REGISTRO EN EL CSV (AQUÍ ESTABA EL ERROR) ---
        registrar_prediccion(
            fuente="Spotify",
            cancion=cancion,
            artista=artista,
            genero=genero_detectado,
            confianza=confianza,
            scores_individuales=resultados  # Le pasamos los resultados de los 3 puntos
        )

        # FASE 5: Recomendación Nandi
        print(f"\n--- Recomendación Nandi ---")
        sugerencia = obtener_playlist_por_genero(genero_detectado)
        if sugerencia:
            print(f" Si te gusta el {genero_detectado}, probá con: {sugerencia['nombre']}")
            print(f" Link: {sugerencia['url']}")
        
    except Exception as e:
        print(f" Error crítico en la predicción: {e}")

if __name__ == "__main__":
    # Probamos con un link real para validar el sistema
    LINK_PRUEBA = "https://open.spotify.com/intl-es/track/41sGGCCoHI2GLV9qadX80A?si=b3f1e1262b4f4cd3" 
    ejecutar_nandi_completo(LINK_PRUEBA)