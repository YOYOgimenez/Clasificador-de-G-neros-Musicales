import librosa
import numpy as np

# Forma fija del espectrograma que espera la CNN
N_MELS   = 128
N_FRAMES = 130
SR       = 22050
DURATION = 3          # segundos por ventana


def preparar_audio_nandi(ruta_audio, offset=0):
    """
    Motor unificado de procesamiento de audio.

    Genera un log-mel espectrograma normalizado listo para la CNN.
    Soporta 'offset' para analizar distintos puntos de la canción.

    Returns:
        np.ndarray de forma (1, N_MELS, N_FRAMES, 1)  →  listo para Keras
        None si el archivo no se puede procesar.
    """
    try:
        # Cargamos la ventana de audio
        y, sr = librosa.load(ruta_audio, sr=SR, duration=DURATION, offset=offset)

        # Mínimo de muestras para un espectrograma útil (0.5 s)
        if len(y) < sr * 0.5:
            print(f"⚠️  Audio demasiado corto en offset={offset}: {ruta_audio}")
            return None

        # --- Espectrograma ---
        S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                               hop_length=512, n_fft=2048)
        S_dB = librosa.power_to_db(S, ref=np.max)   # rango aprox. [-80, 0] dB

        # Ajuste exacto a N_FRAMES columnas
        if S_dB.shape[1] >= N_FRAMES:
            S_dB = S_dB[:, :N_FRAMES]
        else:
            pad = N_FRAMES - S_dB.shape[1]
            S_dB = np.pad(S_dB, ((0, 0), (0, pad)), mode='constant',
                          constant_values=S_dB.min())   # padding con el mínimo, no con 0

        # --- NORMALIZACIÓN (fix principal del no-aprendizaje) ---
        # Min-max por espectrograma → valores en [0, 1]
        s_min, s_max = S_dB.min(), S_dB.max()
        if s_max - s_min > 1e-6:                        # evitamos división por cero
            S_dB = (S_dB - s_min) / (s_max - s_min)
        else:
            S_dB = np.zeros_like(S_dB)                  # silencio total

        # Forma (1, 128, 130, 1) lista para Keras
        return S_dB.reshape(1, N_MELS, N_FRAMES, 1)

    except Exception as e:
        print(f"⚠️  Error en nandi_utils (offset={offset}) — {ruta_audio}: {e}")
        return None
