import os
import random
import json
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(r"C:\Users\joel_\OneDrive\Desktop\music_genre_classification")
DATASET_PATH = BASE_DIR / "datasets" / "Data" / "genres_original"
OUTPUT_PATH = BASE_DIR / "metadata"

# Proporciones (80% entrenamiento, 10% validación, 10% testeo)
TRAIN_PCT = 0.8
VAL_PCT = 0.1

def create_stratified_split():
    # Diccionarios para guardar los caminos de los archivos
    splits = {"train": [], "val": [], "test": []}
    
    # Listamos las carpetas de géneros
    genres = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(DATASET_PATH / d)]
    
    for genre in genres:
        genre_path = DATASET_PATH / genre
        # Obtenemos todos los .wav de este género específico
        files = [f"{genre}/{f}" for f in os.listdir(genre_path) if f.endswith(".wav")]
        
        # Mezclamos solo los archivos de ESTE género
        random.seed(42) # Para que siempre nos de el mismo resultado
        random.shuffle(files)
        
        # Calculamos los puntos de corte
        num_files = len(files)
        train_idx = int(num_files * TRAIN_PCT)
        val_idx = int(num_files * (TRAIN_PCT + VAL_PCT))
        
        # Repartimos de forma exacta
        splits["train"].extend(files[0:train_idx])
        splits["val"].extend(files[train_idx:val_idx])
        splits["test"].extend(files[val_idx:])
        
        print(f"Género '{genre}': {len(files)} archivos repartidos con éxito.")

    # Guardamos los 3 archivos JSON
    OUTPUT_PATH.mkdir(exist_ok=True)
    for name, data in splits.items():
        with open(OUTPUT_PATH / f"{name}.json", "w") as f:
            json.dump(data, f, indent=4)
            
    print("\n Revisá la carpeta 'metadata'.")

if __name__ == "__main__":
    create_stratified_split()
    