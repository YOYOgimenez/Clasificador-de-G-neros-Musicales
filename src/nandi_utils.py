import librosa
import numpy as np

def preparar_audio_nandi(ruta_audio):
    """
    Motor unificado: transforma cualquier audio al formato 
    que tu modelo de Red Neuronal entiende (128x130).
    """
    try:
        # 1. Cargar solo 3 segundos (lo que el modelo espera)
        # Forzamos SR a 22050 para que siempre sea igual
        y, sr = librosa.load(ruta_audio, sr=22050, duration=3)
        
        # 2. Espectrograma de Mel
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        
        # 3. Convertir a Decibelios (SIN normalizaciones extras)
        S_dB = librosa.power_to_db(S)
        
        # 4. Ajuste de ancho a 130 columnas
        if S_dB.shape[1] > 130:
            S_dB = S_dB[:, :130]
        else:
            S_dB = np.pad(S_dB, ((0, 0), (0, 130 - S_dB.shape[1])), mode='constant')
            
        # 5. Formato Final: (1, 128, 130, 1)
        return S_dB.reshape(1, 128, 130, 1)
    except Exception as e:
        print(f"❌ Error en el motor de audio: {e}")
        return None