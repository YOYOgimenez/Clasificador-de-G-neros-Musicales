import os
import librosa
import numpy as np
import json
from pathlib import Path

# --- CONFIGURACIÓN ---
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # segundos por canción en GTZAN
NUM_SEGMENTS = 10    # Dividimos cada canción en 10 partes de 3 seg
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

# Rutas 
BASE_DIR = Path(r"C:\Users\joel_\OneDrive\Desktop\music_genre_classification")
DATASET_PATH = BASE_DIR / "datasets" / "Data" / "genres_original"
JSON_PATH = BASE_DIR / "metadata"
OUTPUT_PATH = BASE_DIR / "processed_data"

def process_split(split_name):
    print(f"\n--- Iniciando transformación de: {split_name} ---")
    
    # Leer el mapa (JSON)
    json_file = JSON_PATH / f"{split_name}.json"
    with open(json_file, "r") as f:
        files_to_process = json.load(f)

    for rel_path in files_to_process:
        # rel_path es algo como "blues/blues.00000.wav"
        # En lugar de separar a mano, usamos os.path.split
        genre, file_name = os.path.split(rel_path)
        input_file = DATASET_PATH / rel_path

        # Crear carpeta de salida: processed_data/train/blues/
        dest_dir = OUTPUT_PATH / split_name / genre
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Cargar el audio
            signal, sr = librosa.load(input_file, sr=SAMPLE_RATE)

            # 2. Cortar en segmentos y extraer características
            for s in range(NUM_SEGMENTS):
                start = SAMPLES_PER_SEGMENT * s
                finish = start + SAMPLES_PER_SEGMENT
                
                # Extraer Espectrograma de Mel
                mel_spec = librosa.feature.melspectrogram(y=signal[start:finish], 
                                                         sr=sr, 
                                                         n_mels=128)
                # Convertir a escala Logarítmica (como escucha el oído humano)
                log_mel_spec = librosa.power_to_db(mel_spec)

                # 3. Guardar como matriz de numpy (.npy)
                save_name = f"{file_name.replace('.wav', '')}_seg{s}.npy"
                np.save(dest_dir / save_name, log_mel_spec)

            print(f"OK: {file_name}")

        except Exception as e:
            print(f"Error procesando {file_name}: {e}")

if __name__ == "__main__":
    # Creamos la carpeta base si no existe
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # Procesamos los 3 grupos
    for split in ["train", "val", "test"]:
        process_split(split)
        
    print("\n Los Datos están listos en 'processed_data'") 