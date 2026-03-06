# 📡 DroneRF Analyzer

Plataforma para análisis de huellas RF de drones mediante espectrogramas STFT y clasificación con redes CNN.

## Características

- **Carga de señales RF**: Soporta CSV, NPY, NPZ, WAV, JSON, TXT, BIN/RAW/IQ
- **STFT (Transformada de Fourier de Ventana Corta)**: Genera espectrogramas con ventanas configurables (Hann, Hamming, Blackman, Kaiser)
- **Etiquetado**: Clasifica por marca de dron, tipo y huella RF. Importa/exporta CSV y JSON
- **Dataset**: Empaqueta espectrogramas etiquetados en ZIP listos para entrenar
- **CNN + Validación Cruzada**: Entrena modelo convolucional con K-Fold Cross-Validation estratificado
- **Métricas**: Accuracy, Precision, Recall, F1-Score, Matriz de Confusión, Curvas de Entrenamiento
- **Exportación**: Descarga modelo entrenado, reporte de resultados y dataset

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/DTL-DA/DroneRF-Analyzer.git
cd DroneRF-Analyzer

# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app.py
```

Se abrirá en `http://localhost:8501`

## Flujo de Trabajo

1. **Cargar y Transformar**: Sube archivos con señales RF → genera espectrogramas STFT
2. **Etiquetar**: Asigna marca, tipo e ID de huella RF a cada espectrograma
3. **Dataset**: Visualiza distribución de clases y descarga el dataset en ZIP
4. **Entrenar CNN**: Configura K-Folds, épocas y batch size → entrena y evalúa

## Arquitectura CNN

- 3 bloques convolucionales (Conv2D + BatchNorm + MaxPool + Dropout)
- Global Average Pooling
- Dense layers (256 → 128 → N clases)
- Optimizador: Adam con ReduceLROnPlateau
- Early Stopping para evitar overfitting

## Formatos de Señal Soportados

| Formato | Descripción                                      |
| ------- | ------------------------------------------------ |
| CSV     | Columnas: valores de señal, o I/Q en 2 columnas  |
| NPY     | Array NumPy 1D (real) o 2D (I/Q)                 |
| WAV     | Audio/señal con frecuencia de muestreo           |
| JSON    | `{signal: [...], fs: ...}` o `{iq: [[I,Q],...]}` |
| BIN/RAW | Float32 intercalado (I,Q,I,Q,...)                |
| TXT     | Valores separados por comas o tabs               |
