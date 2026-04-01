import csv
import os
import numpy as np
from datetime import datetime

# RUTA: Siempre relativa a la raíz del proyecto
HISTORIAL_CSV = "reports/nandi_history.csv"

# Definimos los géneros fijos para asegurar consistencia
GENRES_LIST = ['blues', 'classical', 'country', 'disco', 'hiphop', 
               'jazz', 'metal', 'pop', 'reggae', 'rock']

def inicializar_historial():
    """ Crea la carpeta 'reports' y el CSV si no existen """
    if not os.path.exists('reports'):
        os.makedirs('reports')
        print("📁 Carpeta /reports creada.")
        
    if not os.path.exists(HISTORIAL_CSV):
        headers = ['timestamp', 'fuente', 'cancion', 'artista', 
                   'genero_predicho', 'confianza', 
                   'segundo_10', 'segundo_25', 'segundo_45']
        with open(HISTORIAL_CSV, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        print(f"📄 Archivo CSV inicializado en: {HISTORIAL_CSV}")

def registrar_prediccion(fuente, cancion, artista, genero, confianza, scores_individuales):
    """ Guarda los datos en el CSV de reportes de forma robusta """
    print(f"DEBUG: Intentando registrar en CSV... Fuente: {fuente}")
    
    try:
        inicializar_historial()
        ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Buscamos el índice del género detectado
        genero_lower = genero.lower()
        if genero_lower in GENRES_LIST:
            idx = GENRES_LIST.index(genero_lower)
        else:
            idx = 0 # Fallback por si hay un género extraño

        # 2. Manejo de scores para los 3 fragmentos (Evita el error de índice)
        # Verificamos si scores_individuales es una lista de listas (fragmentos) o plana
        if isinstance(scores_individuales, (list, np.ndarray)) and len(scores_individuales) > 0:
            if isinstance(scores_individuales[0], (list, np.ndarray)):
                # Caso ideal: Tenemos los fragmentos separados
                s10 = scores_individuales[0][idx] * 100 if len(scores_individuales) > 0 else 0
                s25 = scores_individuales[1][idx] * 100 if len(scores_individuales) > 1 else 0
                s45 = scores_individuales[2][idx] * 100 if len(scores_individuales) > 2 else 0
            else:
                # Caso: Tenemos una lista simple (promedio)
                s10 = s25 = s45 = confianza
        else:
            s10 = s25 = s45 = confianza

        # 3. Armamos la fila asegurando que confianza sea un número formateado
        fila = [
            ahora, 
            fuente, 
            cancion, 
            artista, 
            genero.upper(), 
            f"{float(confianza):.2f}", 
            f"{float(s10):.2f}", 
            f"{float(s25):.2f}", 
            f"{float(s45):.2f}"
        ]
        
        with open(HISTORIAL_CSV, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(fila)
            
        print(f"✅ Datos guardados exitosamente en /reports.")
        
    except Exception as e:
        print(f"❌ Error al guardar en CSV: {e}")