import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/v2_improved_model_best.h5"
TEST_DATA_PATH = r"C:\Users\joel_\OneDrive\Desktop\music_genre_classification\processed_data\test"

def run_final_test():
    print("--- INICIANDO EVALUACION FINAL DEL MODELO ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    
    X_test = []
    y_test_labels = []
    EXPECTED_SHAPE = (128, 130) # El tamaño que tu modelo espera
    
    genres = sorted(os.listdir(TEST_DATA_PATH))
    print("Cargando y normalizando datos de prueba...")

    for genre in genres:
        genre_dir = os.path.join(TEST_DATA_PATH, genre)
        for npy_file in os.listdir(genre_dir):
            data = np.load(os.path.join(genre_dir, npy_file))
            
            # --- CORRECCIÓN DE TAMAÑO ---
            # Si el archivo es más grande, lo recortamos. Si es más chico, le ponemos ceros.
            if data.shape[1] > EXPECTED_SHAPE[1]:
                data = data[:, :EXPECTED_SHAPE[1]]
            elif data.shape[1] < EXPECTED_SHAPE[1]:
                pad_width = EXPECTED_SHAPE[1] - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            
            # Si después de eso tiene el tamaño correcto, lo agregamos
            if data.shape == EXPECTED_SHAPE:
                X_test.append(data[..., np.newaxis])
                y_test_labels.append(genre)

    # Ahora sí, el array va a ser homogéneo y no va a tirar error
    X_test = np.array(X_test)
    
    le = LabelEncoder()
    y_true = le.fit_transform(y_test_labels)
    
    print(f"Evaluando {len(X_test)} archivos válidos...")
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    final_acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("       RESULTADOS ACADÉMICOS")
    print("="*40)
    print(f" PRECISIÓN REAL EN TEST: {final_acc * 100:.2f}%")
    print(f" TOTAL ARCHIVOS:        {len(X_test)}")
    print("="*40)

if __name__ == "__main__":
    run_final_test()