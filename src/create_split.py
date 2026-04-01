import os
import json
import random
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATASET_PATH = BASE_DIR / "datasets" / "Data" / "genres_v4"
JSON_PATH    = BASE_DIR / "metadata"

# Semilla fija: los splits serán siempre idénticos entre corridas
RANDOM_SEED = 42

# Orden estricto compartido con train_model.py
GENRES_STRICT = ['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock']


def create_splits(train_pct=0.8, val_pct=0.1):
    if not DATASET_PATH.exists():
        print(f"❌ Error: No se encuentra {DATASET_PATH}. Corré primero augment_v4.py")
        return

    JSON_PATH.mkdir(exist_ok=True)

    # Limpiamos JSONs viejos para evitar contaminación
    for old_json in ["train.json", "val.json", "test.json"]:
        old_path = JSON_PATH / old_json
        if old_path.exists():
            old_path.unlink()
            print(f"🧹 Eliminado {old_json} viejo.")

    data = {"train": [], "val": [], "test": []}

    # Solo procesamos géneros del orden estricto que existan en disco
    genres_found   = [g for g in GENRES_STRICT if (DATASET_PATH / g).is_dir()]
    genres_missing = set(GENRES_STRICT) - set(genres_found)
    if genres_missing:
        print(f"⚠️  Géneros no encontrados en disco: {sorted(genres_missing)}")

    print(f"🎸 Generando splits para: {genres_found}\n")

    rng = random.Random(RANDOM_SEED)  # RNG aislado → reproducible sin tocar random global

    stats = []
    for genre in genres_found:
        genre_path = DATASET_PATH / genre
        files = [f"{genre}/{f}" for f in os.listdir(genre_path) if f.endswith(".wav")]

        if not files:
            print(f"⚠️  Sin archivos .wav en: {genre}")
            continue

        rng.shuffle(files)

        n       = len(files)
        n_train = int(n * train_pct)
        n_val   = int(n * val_pct)
        n_test  = n - n_train - n_val   # el resto va a test (sin pérdida por redondeo)

        data["train"].extend(files[:n_train])
        data["val"].extend(files[n_train : n_train + n_val])
        data["test"].extend(files[n_train + n_val :])

        stats.append((genre, n_train, n_val, n_test))

    # Tabla de resumen
    print(f"{'Género':12s} | {'Train':>6} | {'Val':>5} | {'Test':>5}")
    print("-" * 38)
    for genre, tr, va, te in stats:
        print(f"{genre:12s} | {tr:>6} | {va:>5} | {te:>5}")

    # Mezclamos los splits globales para que el modelo no vea todos los blues juntos
    rng.shuffle(data["train"])
    rng.shuffle(data["val"])
    rng.shuffle(data["test"])

    print()
    for split in ["train", "val", "test"]:
        out_path = JSON_PATH / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(data[split], f, indent=4)
        print(f"✅ {split}.json → {len(data[split])} archivos")

    print("\n✨ Splits listos. Podés correr extract_features.py")


if __name__ == "__main__":
    create_splits()
