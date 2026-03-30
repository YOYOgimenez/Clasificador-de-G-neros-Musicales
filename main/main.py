import os
import sys
from datetime import datetime

# Función para guardar el reporte en un archivo .txt
def save_report(content, report_name="prediction_report"):
    if not os.path.exists("reports"):
        os.makedirs("reports")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/{report_name}_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n[INFO] Reporte guardado en: {filename}")

def menu():
    print("\n" + "="*40)
    print("     SISTEMA DE REPORTES")
    print("="*40)
    print("1. Ejecutar Evaluación Final (Generar Score)")
    print("2. Analizar Archivo MP3 (Generar Predicción)")
    print("3. Salir")
    print("="*40)
    
    opcion = input("Elegí una opción: ")
    
    if opcion == "1":
        from src.final_evaluation import run_final_test
        # Aquí capturamos lo que hace tu test (puedes modificar run_final_test para que retorne el string)
        print("Ejecutando evaluación...")
        run_final_test() 
        # (Opcional: podrías automatizar la escritura del resultado aquí)

    elif opcion == "2":
        from src.predict_real_time import predict_genre
        import tensorflow as tf
        
        MODEL_PATH = "models/v2_improved_model_best.h5"
        FILE_PATH = input("Ruta del MP3: ").strip().replace('"', '')
        
        if os.path.exists(FILE_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            # Aquí llamamos a la predicción
            
            predict_genre(FILE_PATH, model)
            
            # Ejemplo de cómo se guardaría el reporte manual
            report_text = f"REPORTE DE PREDICCIÓN DEL MODELO \n"
            report_text += f"Fecha: {datetime.now()}\n"
            report_text += f"Archivo: {os.path.basename(FILE_PATH)}\n"
            report_text += "-"*30 + "\n"
            save_report(report_text, "prediccion_individual")
        else:
            print("Error: Archivo no encontrado.")

    elif opcion == "3":
        sys.exit()

if __name__ == "__main__":
    menu()