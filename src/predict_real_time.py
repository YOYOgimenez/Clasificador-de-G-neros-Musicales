import os
import librosa
import numpy as np
import tensorflow as tf

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/v2_improved_model_best.h5"
FILE_TO_PREDICT = r"C:\Users\joel_\Downloads\cancion_de_yt.mp3" 

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def predict_genre(file_path, model):
    print(f"\n[INFO] Analizando archivo: {os.path.basename(file_path)}")
    
    try:
        # 1. Cargar audio (Empezamos en el segundo 45 para saltar la intro)
        y, sr = librosa.load(file_path, offset=45, duration=30)
        
        # 2. Espectrograma de Mel
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 3. Normalización Min-Max (Clave para que no se maree con el volumen)
        min_val = np.min(mel_spec_db)
        max_val = np.max(mel_spec_db)
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val + 1e-6)
        
        # 4. Ajustar tamaño a (128, 130)
        if mel_spec_norm.shape[1] > 130:
            mel_spec_norm = mel_spec_norm[:, :130]
        else:
            pad_width = 130 - mel_spec_norm.shape[1]
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')

        # 5. Formatear para el modelo (1, 128, 130, 1)
        input_data = mel_spec_norm[np.newaxis, ..., np.newaxis]

        # 6. Predicción
        predictions = model.predict(input_data, verbose=0)
        
        # 7. REPORTE DE RESULTADOS
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx] * 100
        
        print("\n" + "="*40)
        print("       ANÁLISIS DEL MODELO")
        print("="*40)
        print(f" GÉNERO PREDICHO: {GENRES[predicted_idx].upper()}")
        print(f" CONFIANZA:       {confidence:.2f}%")
        print("="*40)
        
        print("\nProbabilidades por categoría:")
        # Ordenamos de mayor a menor probabilidad
        sorted_indices = np.argsort(predictions[0])[::-1]
        for i in sorted_indices:
            prob = predictions[0][i] * 100
            if prob > 0.1: # Solo mostramos los que tienen algo de chance
                print(f" - {GENRES[i].capitalize()}: {prob:.2f}%")

    except Exception as e:
        print(f"\n[ERROR] No se pudo procesar el audio: {e}")

if __name__ == "__main__":
    if os.path.exists(FILE_TO_PREDICT):
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            predict_genre(FILE_TO_PREDICT, model)
        else:
            print(f"[ERROR] No se encontró el modelo en {MODEL_PATH}")
    else:
        print(f"[ERROR] El archivo de audio no existe en: {FILE_TO_PREDICT}")