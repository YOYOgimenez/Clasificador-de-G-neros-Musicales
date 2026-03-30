import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
HISTORY_PATH = "models/v2_improved_model_history.csv"
OUTPUT_IMAGE = "reports/v2_training_curves.png"

def plot_history(csv_path):
    # Crear carpeta de reportes si no existe
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    # Leer el diario de viaje del modelo
    data = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 5))

    # Gráfico 1: Precisión (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(data['accuracy'], label='Entrenamiento (Train)')
    plt.plot(data['val_accuracy'], label='Validación (Val)')
    plt.title('Evolución de la Precisión')
    plt.xlabel('Época')
    plt.ylabel('Exactitud (0.0 - 1.0)')
    plt.legend()
    plt.grid(True)

    # Gráfico 2: Error (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(data['loss'], label='Entrenamiento (Train)')
    plt.plot(data['val_loss'], label='Validación (Val)')
    plt.title('Evolución del Error (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Valor de Pérdida')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    plt.show()
    print(f"Gráfico guardado en: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    if os.path.exists(HISTORY_PATH):
        plot_history(HISTORY_PATH)
    else:
        print(f"No se encontró el archivo {HISTORY_PATH}. ¿Corrio el entrenamiento V2?")