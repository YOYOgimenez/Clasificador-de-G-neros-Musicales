import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURACIÓN ---
DATA_PATH = r"C:\Users\joel_\OneDrive\Desktop\music_genre_classification\processed_data"
MODEL_NAME = "v2_improved_model" # prueba 2 

def load_data_from_dir(base_path):
    X, y = [], []
    genres = sorted(os.listdir(base_path))
    EXPECTED_SHAPE = (128, 130)
    for genre in genres:
        genre_dir = os.path.join(base_path, genre)
        for npy_file in os.listdir(genre_dir):
            data = np.load(os.path.join(genre_dir, npy_file))
            if data.shape == EXPECTED_SHAPE:
                X.append(data[..., np.newaxis])
                y.append(genre)
    return np.array(X), np.array(y)

def build_improved_model(input_shape, num_classes):
    model = models.Sequential([
        # Capa 1: Detecta bordes y sonidos simples
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Capa 2: Detecta patrones rítmicos
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        # Capa 3 NUEVA: Detecta estructuras más complejas
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        # Aumentamos a 128 neuronas para darle más "capacidad de razonamiento"
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Subimos el Dropout para evitar que se machetee (Overfitting)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X_train, y_train_raw = load_data_from_dir(os.path.join(DATA_PATH, "train"))
    X_val, y_val_raw = load_data_from_dir(os.path.join(DATA_PATH, "val"))

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    model = build_improved_model(X_train.shape[1:], len(le.classes_))

    # --- LOS GUARDIANES DEL ENTRENAMIENTO ---
    checkpoint = callbacks.ModelCheckpoint(
        f"models/{MODEL_NAME}_best.h5", 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, # Si por 5 vueltas no mejora, se corta solo
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=50, # Tiramos 50 porque el EarlyStop nos cuida
        batch_size=32,
        callbacks=[checkpoint, early_stop]
    )

    # Guardamos el historial en un CSV para comparar después
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f"models/{MODEL_NAME}_history.csv", index=False)
    print(f"\nEntrenamiento terminado. Historial guardado en models/{MODEL_NAME}_history.csv")