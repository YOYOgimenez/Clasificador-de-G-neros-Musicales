import csv
import os
from datetime import datetime

# RUTA: Siempre relativa a la raíz del proyecto
HISTORIAL_CSV = "reports/nandi_history.csv"

# Definimos los géneros acá para evitar el "Import Circular"
GENRES_LIST = ['blues', 'classical', 'country', 'disco', 'hiphop', 
               'jazz', 'metal', 'pop', 'reggae', 'rock']

def inicializar_historial():
    """ Crea la carpeta 'reports' y el CSV si no existen """
    if not os.path.exists('reports'):
        os.makedirs('reports')
        print(" Carpeta /reports creada.")
        
    if not os.path.exists(HISTORIAL_CSV):
        headers = ['timestamp', 'fuente', 'cancion', 'artista', 
                   'genero_predicho', 'confianza', 
                   'segundo_10', 'segundo_25', 'segundo_45']
        with open(HISTORIAL_CSV, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        print(f" Archivo CSV inicializado en: {HISTORIAL_CSV}")

def registrar_prediccion(fuente, cancion, artista, genero, confianza, scores_individuales):
    """ Guarda los datos en el CSV de reportes """
    print(f"DEBUG: Intentando registrar en CSV... Fuente: {fuente}")
    
    try:
        inicializar_historial()
        
        ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Usamos la lista local para evitar imports circulares
        idx = GENRES_LIST.index(genero.lower())
        
        # Extraemos los scores de cada fragmento
        s10 = scores_individuales[0][idx] * 100
        s25 = scores_individuales[1][idx] * 100
        s45 = scores_individuales[2][idx] * 100

        fila = [ahora, fuente, cancion, artista, genero.upper(), 
                f"{confianza:.2f}", f"{s10:.2f}", f"{s25:.2f}", f"{s45:.2f}"]
        
        with open(HISTORIAL_CSV, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(fila)
            
        print(f" Datos guardados exitosamente en /reports.")
        
    except Exception as e:
        print(f" Error al guardar en CSV: {e}")