# DronRForensic

**App para recopilación forense de señales RF de drones e identificación mediante CNN.**

## Descripción

DronRForensic es una aplicación web construida con Streamlit que permite:

1. **Cargar señales RF** de drones (CSV, NumPy NPY, binario IQ float32, WAV estéreo).
2. **Generar espectrogramas** mediante Transformada de Fourier de Ventana Corta (STFT).
3. **Etiquetar y gestionar** un dataset de espectrogramas con marca y tipo de dron.
4. **Exportar el dataset** en formato HDF5 o NPZ para reutilización.
5. **Entrenar un modelo CNN** con validación cruzada K-fold estratificada.
6. **Identificar drones desconocidos** a partir de su huella de RF.

---

## Instalación

```bash
git clone <repo>
cd DronRForensic
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Abre el navegador en `http://localhost:8501`.

---

## Estructura del Proyecto

```
DronRForensic/
├── app.py                     # Aplicación Streamlit principal
├── requirements.txt           # Dependencias Python
├── src/
│   ├── signal_processor.py    # Carga de señales + STFT + espectrogramas
│   ├── dataset_handler.py     # Gestión, etiquetado y serialización del dataset
│   ├── model.py               # Arquitectura CNN (Keras/TensorFlow)
│   └── trainer.py             # Validación cruzada K-fold + entrenamiento
└── tests/
    ├── test_signal_processor.py
    └── test_dataset_handler.py
```

---

## Flujo de Trabajo

### 1. Procesamiento STFT
- Configura los parámetros STFT (fs, nperseg, NFFT, ventana).
- Carga un archivo de señal o genera una señal sintética.
- Visualiza el espectrograma dB y el normalizado.
- Etiqueta la señal (marca + modelo) y agrégala al dataset.

### 2. Gestión del Dataset
- Carga datasets previos en formato HDF5 o NPZ.
- Visualiza la distribución de clases y ejemplos de espectrogramas.
- Descarga el dataset actualizado.

### 3. Entrenamiento CNN
- Configura folds (K), épocas y batch size.
- Ejecuta validación cruzada estratificada.
- Visualiza precisión/pérdida por fold y matriz de confusión.
- Entrena el modelo final y descárgalo como `.h5`.

### 4. Identificación
- Carga una señal desconocida o genera una sintética.
- El modelo devuelve marca, tipo y probabilidad de confianza.

---

## Formatos de Señal Soportados

| Formato | Descripción |
|---------|-------------|
| `.csv`  | Columnas `I` y `Q` (o dos primeras columnas) |
| `.npy`  | Array NumPy complejo o de forma `(N, 2)` |
| `.bin` / `.dat` | Datos IQ binarios intercalados float32 |
| `.wav`  | WAV mono (amplitud) o estéreo (I/Q) |

---

## Arquitectura CNN

```
Input (H × W × 1)
 └─ Conv2D(32) × 2 + BatchNorm + MaxPool + Dropout(0.25)
 └─ Conv2D(64) × 2 + BatchNorm + MaxPool + Dropout(0.25)
 └─ Conv2D(128) × 2 + BatchNorm + MaxPool + Dropout(0.25)
 └─ GlobalAveragePooling2D
 └─ Dense(256) + Dropout(0.5)
 └─ Dense(n_classes, softmax)
```

Optimizador: Adam (lr=1e-3) · Pérdida: Sparse Categorical Crossentropy

---

## Ejecución de Tests

```bash
pytest tests/ -v
```

---

## Licencia

© 2024 DTL-DA
