import librosa
import soundfile as sf
import numpy as np
import os

# --- CONFIGURACIÓN ---
ORIGINAL_DIR = "datasets/Data/genres_original"
OUTPUT_DIR   = "datasets/Data/genres_v4"

# Semilla para reproducibilidad
SEED = 42
rng  = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Transformaciones de augmentación
# ---------------------------------------------------------------------------

def pitch_shift(y, sr):
    """Sube o baja el tono entre -3 y +3 semitonos."""
    n_steps = rng.choice([-3, -2, 2, 3])
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=float(n_steps))

def time_stretch(y):
    """Estira o comprime el tiempo entre 0.85x y 1.15x sin cambiar el tono."""
    rate = rng.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(y, rate=rate)

def add_noise(y):
    """Agrega ruido gaussiano suave (simula grabaciones en distintos ambientes)."""
    noise_level = rng.uniform(0.003, 0.008)
    noise = rng.normal(0, noise_level, len(y))
    return np.clip(y + noise, -1.0, 1.0)

def volume_shift(y):
    """Sube o baja el volumen entre 0.7x y 1.3x."""
    factor = rng.uniform(0.7, 1.3)
    return np.clip(y * factor, -1.0, 1.0)

def random_crop_pad(y, sr, target_duration=30):
    """
    Toma un fragmento aleatorio de la canción.
    Si el audio es más corto que target_duration lo rellena.
    """
    target_len = sr * target_duration
    if len(y) >= target_len:
        max_start = len(y) - target_len
        start = rng.integers(0, max_start + 1)
        return y[start : start + target_len]
    else:
        pad = target_len - len(y)
        return np.pad(y, (0, pad), mode='constant')


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

AUGMENTATIONS = {
    "orig":    lambda y, sr: y,                          # 1. Original sin cambios
    "pitch":   lambda y, sr: pitch_shift(y, sr),         # 2. Pitch shift
    "stretch": lambda y, sr: time_stretch(y),            # 3. Time stretch
    "noise":   lambda y, sr: add_noise(y),               # 4. Ruido suave
    "volume":  lambda y, sr: volume_shift(y),            # 5. Volumen distinto
    "crop":    lambda y, sr: random_crop_pad(y, sr),     # 6. Fragmento aleatorio
}
# → 6 versiones × 100 archivos = 600 por género → 6000 total


def augment_v4():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_ok  = 0
    total_err = 0

    for genre in sorted(os.listdir(ORIGINAL_DIR)):
        genre_path     = os.path.join(ORIGINAL_DIR, genre)
        out_genre_path = os.path.join(OUTPUT_DIR, genre)

        if not os.path.isdir(genre_path):
            continue

        os.makedirs(out_genre_path, exist_ok=True)

        wav_files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
        print(f"\n📦 {genre} ({len(wav_files)} archivos originales)...")

        genre_ok = 0
        for file in wav_files:
            file_path = os.path.join(genre_path, file)
            base_name = file.replace(".wav", "")

            try:
                y, sr = librosa.load(file_path, sr=22050)

                for aug_name, aug_fn in AUGMENTATIONS.items():
                    try:
                        y_aug     = aug_fn(y, sr)
                        out_name  = f"{aug_name}_{base_name}.wav"
                        out_path  = os.path.join(out_genre_path, out_name)
                        sf.write(out_path, y_aug, sr)
                        genre_ok += 1
                    except Exception as e:
                        print(f"\n   ⚠️  {aug_name} falló en {file}: {e}")
                        total_err += 1

            except Exception as e:
                print(f"\n   ⚠️  No se pudo cargar {file}: {e}")
                total_err += 1
                continue

        total_ok += genre_ok
        print(f"   ✅ {genre_ok} archivos generados")

    print(f"\n✨ Dataset V4 listo — {total_ok} archivos generados, {total_err} errores.")
    print(f"   Promedio por género: {total_ok // len(os.listdir(ORIGINAL_DIR))} archivos")
    print(f"\n➡️  Ahora corré: create_split.py → extract_features.py → train_model.py")


if __name__ == "__main__":
    augment_v4()