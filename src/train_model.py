import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR   = Path(os.getcwd())
DATA_PATH  = BASE_DIR / "processed_data"
MODEL_DIR  = BASE_DIR / "models"
MODEL_NAME = "nandi_v4_final"

# Semilla global para reproducibilidad
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Orden estricto compartido con create_split.py
GENRES_STRICT = ['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock']


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def cargar_datos(split_name):
    """Carga todos los .npy de un split en memoria."""
    X, y = [], []
    split_path = DATA_PATH / split_name
    found_genres = []

    print(f"📦 Cargando {split_name}...")

    for idx, genre in enumerate(GENRES_STRICT):
        genre_dir = split_path / genre
        if not genre_dir.exists():
            print(f"   ⚠️  Carpeta no encontrada: {genre}")
            continue

        files = [f for f in os.listdir(genre_dir) if f.endswith(".npy")]
        if not files:
            print(f"   ⚠️  Sin archivos en: {genre}")
            continue

        for npy_file in files:
            data = np.load(genre_dir / npy_file)   # (128, 130, 1) — ya normalizado
            X.append(data)
            y.append(idx)

        found_genres.append((genre, len(files)))

    for genre, count in found_genres:
        print(f"   {genre:12s} → {count} archivos")

    if not X:
        return np.array([]), np.array([])

    X = np.array(X, dtype=np.float32)   # (N, 128, 130, 1)
    y = np.array(y, dtype=np.int32)
    print(f"   Shape final: {X.shape}  |  Clases: {np.unique(y)}\n")
    return X, y


# ---------------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------------

def build_model(input_shape, num_classes):
    """
    CNN con BatchNorm + L2 regularization.
    Más estable que la versión anterior para datasets pequeños.
    """
    reg = regularizers.l2(1e-4)

    model = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Clasificador
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name="nandi_v4")

    return model


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Cargar datos
    X_train, y_train = cargar_datos("train")
    X_val,   y_val   = cargar_datos("val")

    if len(X_train) == 0:
        print("❌ ERROR: No se cargaron datos de entrenamiento. Revisá 'processed_data/train'")
        exit(1)

    # 2. Verificación rápida de rangos (los valores deben estar en [0, 1])
    print(f"🔍 Verificación de rangos:")
    print(f"   X_train — min: {X_train.min():.3f}  max: {X_train.max():.3f}  mean: {X_train.mean():.3f}")
    print(f"   X_val   — min: {X_val.min():.3f}  max: {X_val.max():.3f}  mean: {X_val.mean():.3f}\n")

    # 3. Construir modelo
    input_shape = X_train.shape[1:]   # (128, 130, 1)
    model = build_model(input_shape, len(GENRES_STRICT))
    model.summary()

    # 4. Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / f"{MODEL_NAME}_best.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # ReduceLROnPlateau: baja el LR cuando el val_loss se estanca
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,         # LR = LR * 0.5
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # Log de métricas por época en CSV
    csv_logger = callbacks.CSVLogger(
        str(MODEL_DIR / f"{MODEL_NAME}_history.csv"), append=False
    )

    # 6. Entrenar
    print(f"🚀 Entrenando sobre {len(X_train)} muestras | val: {len(X_val)}")
    print(f"   Input shape: {input_shape}  |  Clases: {len(GENRES_STRICT)}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
        shuffle=True
    )

    # 7. Guardar etiquetas
    labels_path = MODEL_DIR / f"{MODEL_NAME}_labels.txt"
    with open(labels_path, "w") as f:
        f.write("\n".join(GENRES_STRICT))
    print(f"\n✅ Etiquetas guardadas en {labels_path}")

    # 8. Resumen final
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    best_epoch   = history.history.get('val_accuracy', [0]).index(best_val_acc) + 1
    print(f"\n🏆 Mejor val_accuracy: {best_val_acc:.4f} (época {best_epoch})")
    print(f"✨ Proceso terminado. Modelo guardado en {MODEL_DIR}/{MODEL_NAME}_best.h5")
