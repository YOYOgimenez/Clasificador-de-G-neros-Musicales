import os
import librosa
import numpy as np
import tensorflow as tf
from nandi_history import registrar_prediccion 

# --- CONFIGURACIÓN ---
# Usamos una ruta que funcione si lo lanzas desde la raíz o desde src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "v2_improved_model_best.h5")

FILE_TO_PREDICT = r"C:\Users\joel_\Downloads\cancion_de_yt.mp3" 

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def extraer_segmento(file_path, offset_seg):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3, offset=offset_seg)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S)
        
        if S_dB.shape[1] > 130:
            S_dB = S_dB[:, :130]
        else:
            S_dB = np.pad(S_dB, ((0, 0), (0, 130 - S_dB.shape[1])), mode='constant')
            
        return S_dB.reshape(1, 128, 130, 1)
    except Exception as e:
        print(f" Error extrayendo en segundo {offset_seg}: {e}")
        return None

def predict_genre_local_pro(file_path):
    if not os.path.exists(file_path):
        print(f" El archivo no existe en: {file_path}")
        return

    print(f"\n Nandi AI iniciando análisis profundo de: {os.path.basename(file_path)}")

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        
        puntos_de_analisis = [10, 25, 45]
        predicciones_acumuladas = []

        for p in puntos_de_analisis:
            feat = extraer_segmento(file_path, p)
            if feat is not None:
                preds = model.predict(feat, verbose=0)
                predicciones_acumuladas.append(preds[0])

        if not predicciones_acumuladas:
            print(" No se pudo procesar ningún fragmento del audio.")
            return

        promedio_final = np.mean(predicciones_acumuladas, axis=0)
        idx_ganador = np.argmax(promedio_final)
        genero_final = GENRES[idx_ganador]
        confianza_final = promedio_final[idx_ganador] * 100

        # --- FASE DE REGISTRO (LA QUE FALTABA) ---
        registrar_prediccion(
            fuente="Local",
            cancion=os.path.basename(file_path),
            artista="Archivo Local",
            genero=genero_final,
            confianza=confianza_final,
            scores_individuales=predicciones_acumuladas
        )

        print("\n" + "--" * 10)
        print(f"  RESULTADO: {genero_final.upper()}")
        print(f"  CONFIANZA: {confianza_final:.2f}%")
        print("--" * 10)

    except Exception as e:
        print(f" Error crítico en la predicción: {e}")

if __name__ == "__main__":
    predict_genre_local_pro(FILE_TO_PREDICT)