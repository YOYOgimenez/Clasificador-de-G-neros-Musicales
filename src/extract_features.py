import os
import shutil
import numpy as np
import json
from pathlib import Path
from nandi_utils import preparar_audio_nandi, N_MELS, N_FRAMES

# --- CONFIGURACIÓN ---
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
if not (BASE_DIR / "datasets").exists():
    BASE_DIR = Path(os.getcwd())

DATASET_PATH = BASE_DIR / "datasets" / "Data" / "genres_v4"
JSON_PATH    = BASE_DIR / "metadata"
OUTPUT_PATH  = BASE_DIR / "processed_data"


def process_split(split_name):
    print(f"\n--- 🚀 Extrayendo features de: {split_name.upper()} ---")

    json_file = JSON_PATH / f"{split_name}.json"
    if not json_file.exists():
        print(f"❌ Error: No se encuentra {json_file}. Corré primero create_split.py")
        return 0, 0

    with open(json_file, "r") as f:
        files_to_process = json.load(f)

    # Limpieza de carpeta de salida para no mezclar datos viejos
    split_dir = OUTPUT_PATH / split_name
    if split_dir.exists():
        print(f"🧹 Limpiando datos viejos en {split_dir}...")
        shutil.rmtree(split_dir)

    ok_count  = 0
    err_count = 0
    total     = len(files_to_process)

    for i, rel_path in enumerate(files_to_process, 1):
        genre, file_name = os.path.split(rel_path)
        input_file = DATASET_PATH / rel_path

        if not input_file.exists():
            print(f"\n⚠️  Archivo no encontrado: {input_file}")
            err_count += 1
            continue

        dest_dir = OUTPUT_PATH / split_name / genre
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            spec = preparar_audio_nandi(str(input_file))

            if spec is None:
                err_count += 1
                continue

            # spec tiene forma (1, 128, 130, 1) → guardamos (128, 130, 1)
            # Consistente con lo que carga train_model.py
            feature_matrix = spec[0]   # shape: (N_MELS, N_FRAMES, 1)

            save_name = file_name.replace(".wav", ".npy")
            np.save(dest_dir / save_name, feature_matrix)
            ok_count += 1

        except Exception as e:
            print(f"\n⚠️  Error en {file_name}: {e}")
            err_count += 1

        # Progreso en línea
        if i % 50 == 0 or i == total:
            print(f"   [{i:4d}/{total}] ✅ ok: {ok_count}  ❌ err: {err_count}", end="\r")

    print(f"\n   Completado → ✅ {ok_count} ok  ❌ {err_count} errores")
    return ok_count, err_count


if __name__ == "__main__":
    OUTPUT_PATH.mkdir(exist_ok=True)

    total_ok  = 0
    total_err = 0

    for split in ["train", "val", "test"]:
        ok, err = process_split(split)
        total_ok  += ok
        total_err += err

    print(f"\n✨ Features listos — Total: ✅ {total_ok} guardados  ❌ {total_err} errores")
    if total_err > 0:
        print("   Revisá los ⚠️  de arriba para identificar archivos problemáticos.")
