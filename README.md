# 🎷 Nandi AI: Clasificación de Géneros Musicales en Vivo

**Nandi AI** es una plataforma de Inteligencia Artificial que procesa señales de audio para identificar géneros musicales mediante el análisis de **Espectrogramas de Mel** y **Redes Neuronales Convolucionales (CNN)**.

> *Proyecto desarrollado por **Joel Gimenez** como parte de la trayectoria en la **Licenciatura en Ciencias de Datos (UBA)**.*

---

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](TU_LINK_DE_STREAMLIT_AQUÍ)
*(Una vez desplegado en Streamlit Cloud, pegá el link aquí para que cualquiera pueda probarlo)*

## 🧠 Arquitectura Técnica
El sistema no analiza el archivo de audio directamente, sino que lo transforma en una representación visual de frecuencia y tiempo:

1.  **Procesamiento de Audio:** Conversión de señales temporales a **Espectrogramas de Mel** (frecuencias ajustadas a la percepción auditiva humana).
2.  **Modelo CNN:** Arquitectura de visión artificial optimizada con `Dropout` y `Batch Normalization` para asegurar la generalización.
3.  **Análisis de Segmentos:** La IA evalúa tres fragmentos de la canción (10s, 25s, 45s) y promedia las probabilidades para una predicción más robusta.
4.  **Lógica de Incertidumbre:** Si la confianza es inferior al **50%**, el sistema notifica una "Señal Ambigua", permitiendo entender cuándo el modelo tiene dudas razonables.



## 📈 Performance y Métricas
* **Precisión Global:** 69.40%
* **Fortalezas:** Excelente desempeño en **Metal (87%)** y **Música Clásica (82%)**.
* **Dataset:** GTZAN (1000 muestras, 10 géneros).

### Visualización del Entrenamiento
| Matriz de Confusión | Curvas de Aprendizaje |
| :---: | :---: |
| ![Matriz](reports/confusion_matrix.png) | ![Curvas](reports/v2_training_curves.png) |

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.10+
* **Deep Learning:** TensorFlow / Keras
* **Procesamiento de Audio:** Librosa
* **Interfaz Web:** Streamlit
* **Visualización:** Matplotlib, Seaborn

## 💻 Instalación Local
1. Clonar repositorio: `git clone https://github.com/TU_USUARIO/music_genre_classification.git`
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar Dashboard: `python -m streamlit run app/app_nandi.py`

---
**Contacto:** [LinkedIn](https://www.linkedin.com/in/joelgimenez/) | [Email](mailto:joelgimenezl72@gmail.com)