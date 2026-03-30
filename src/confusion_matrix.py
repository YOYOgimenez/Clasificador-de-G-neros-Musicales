import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURACIÓN ---
DATA_PATH = r"C:\Users\joel_\OneDrive\Desktop\music_genre_classification\processed_data"
MODEL_PATH = "models/v2_improved_model_best.h5"

def load_test_data(test_path):
    X, y = [], []
    genres = sorted(os.listdir(test_path))
    EXPECTED_SHAPE = (128, 130)
    for genre in genres:
        genre_dir = os.path.join(test_path, genre)
        for npy_file in os.listdir(genre_dir):
            data = np.load(os.path.join(genre_dir, npy_file))
            if data.shape == EXPECTED_SHAPE:
                X.append(data[..., np.newaxis])
                y.append(genre)
    return np.array(X), np.array(y), genres

if __name__ == "__main__":
    # 1. Cargar datos de TEST (la prueba final)
    print("Cargando datos de prueba...")
    X_test, y_test_raw, genre_names = load_test_data(os.path.join(DATA_PATH, "test"))

    # 2. Cargar el mejor modelo
    model = tf.keras.models.load_model(MODEL_PATH)

    # 3. Hacer predicciones
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 4. Convertir etiquetas reales a números
    le = LabelEncoder()
    y_true = le.fit_transform(y_test_raw)

    # 5. Crear la Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=genre_names, yticklabels=genre_names)
    plt.xlabel('Predicción del modelo')
    plt.ylabel('Género Real')
    plt.title('Matriz de Confusión')
    
    if not os.path.exists('reports'): os.makedirs('reports')
    plt.savefig('reports/confusion_matrix.png')
    plt.show()

    # 6. Informe detallado por género
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=genre_names))