"""
Módulo CNN y Entrenamiento con Validación Cruzada
Define la arquitectura CNN para clasificación de huellas RF de drones
y el pipeline de entrenamiento con K-Fold Cross-Validation.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import json
import io


def crear_modelo_cnn(input_shape, num_clases):
    """
    Crea un modelo CNN para clasificación de espectrogramas RF de drones.
    
    Arquitectura:
    - 3 bloques convolucionales (Conv2D + BatchNorm + MaxPool + Dropout)
    - Global Average Pooling
    - Dense layers para clasificación
    
    Args:
        input_shape: Forma de entrada (alto, ancho, canales), ej: (128, 128, 1).
        num_clases: Número de clases a clasificar.
    
    Returns:
        modelo compilado de Keras.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    modelo = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Clasificador
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_clases, activation='softmax')
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo


def entrenar_con_validacion_cruzada(X, y, clases, n_folds=5, epochs=50,
                                     batch_size=32, callback_progreso=None):
    """
    Entrena el modelo CNN con K-Fold Stratified Cross-Validation.
    
    Args:
        X: Array de imágenes (N, H, W, C).
        y: Array de labels (N,).
        clases: Lista de nombres de clases.
        n_folds: Número de folds para validación cruzada.
        epochs: Épocas de entrenamiento por fold.
        batch_size: Tamaño de batch.
        callback_progreso: Función callback(fold, epoch, logs) para reportar progreso.
    
    Returns:
        dict con resultados detallados de la validación cruzada.
    """
    import tensorflow as tf

    num_clases = len(clases)
    input_shape = X.shape[1:]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    resultados_folds = []
    todas_predicciones = []
    todos_reales = []
    historiales = []
    mejor_accuracy = 0
    mejor_modelo_weights = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Crear modelo nuevo para cada fold
        modelo = crear_modelo_cnn(input_shape, num_clases)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]

        # Entrenar
        historial = modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        historiales.append({
            'loss': [float(v) for v in historial.history['loss']],
            'accuracy': [float(v) for v in historial.history['accuracy']],
            'val_loss': [float(v) for v in historial.history['val_loss']],
            'val_accuracy': [float(v) for v in historial.history['val_accuracy']],
        })

        # Evaluar
        y_pred = np.argmax(modelo.predict(X_val, verbose=0), axis=1)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        resultado_fold = {
            'fold': fold_idx + 1,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'epochs_ejecutadas': len(historial.history['loss'])
        }
        resultados_folds.append(resultado_fold)

        todas_predicciones.extend(y_pred.tolist())
        todos_reales.extend(y_val.tolist())

        # Guardar mejor modelo
        if acc > mejor_accuracy:
            mejor_accuracy = acc
            mejor_modelo_weights = modelo.get_weights()

        if callback_progreso:
            callback_progreso(fold_idx + 1, n_folds, resultado_fold)

    # Métricas globales
    acc_global = accuracy_score(todos_reales, todas_predicciones)
    reporte = classification_report(
        todos_reales, todas_predicciones,
        target_names=clases, output_dict=True, zero_division=0
    )
    matriz_confusion = confusion_matrix(todos_reales, todas_predicciones).tolist()

    # Reconstruir mejor modelo
    mejor_modelo = crear_modelo_cnn(input_shape, num_clases)
    mejor_modelo.set_weights(mejor_modelo_weights)

    resultado_final = {
        'folds': resultados_folds,
        'accuracy_promedio': float(np.mean([r['accuracy'] for r in resultados_folds])),
        'accuracy_std': float(np.std([r['accuracy'] for r in resultados_folds])),
        'precision_promedio': float(np.mean([r['precision'] for r in resultados_folds])),
        'recall_promedio': float(np.mean([r['recall'] for r in resultados_folds])),
        'f1_promedio': float(np.mean([r['f1_score'] for r in resultados_folds])),
        'accuracy_global': float(acc_global),
        'reporte_clasificacion': reporte,
        'matriz_confusion': matriz_confusion,
        'clases': clases,
        'n_folds': n_folds,
        'total_muestras': len(y),
        'historiales': historiales
    }

    return resultado_final, mejor_modelo


def exportar_modelo(modelo, clases, ruta_base="modelo_dronerf"):
    """
    Exporta el modelo entrenado junto con metadata.
    
    Returns:
        buffer: BytesIO con archivo ZIP del modelo.
    """
    import tensorflow as tf

    buffer = io.BytesIO()
    
    with io.BytesIO() as model_buffer:
        # Guardar modelo en formato SavedModel dentro de un temp dir
        import tempfile, shutil, os
        
        tmpdir = tempfile.mkdtemp()
        modelo_path = os.path.join(tmpdir, "modelo_dronerf")
        modelo.save(modelo_path)
        
        import zipfile
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(modelo_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, tmpdir)
                    zf.writestr(arcname, open(file_path, 'rb').read())
            
            # Metadata
            metadata = {
                'clases': clases,
                'num_clases': len(clases),
                'input_shape': list(modelo.input_shape[1:]),
                'arquitectura': 'CNN DroneRF Analyzer'
            }
            zf.writestr('metadata.json',
                        json.dumps(metadata, indent=2, ensure_ascii=False))
        
        shutil.rmtree(tmpdir)

    buffer.seek(0)
    return buffer


def generar_reporte_texto(resultados):
    """Genera un reporte en texto plano de los resultados de validación cruzada."""
    lineas = [
        "=" * 60,
        "  REPORTE DE VALIDACIÓN CRUZADA - DroneRF Analyzer",
        "=" * 60,
        "",
        f"  Folds: {resultados['n_folds']}",
        f"  Muestras totales: {resultados['total_muestras']}",
        f"  Clases: {', '.join(resultados['clases'])}",
        "",
        "-" * 60,
        "  RESULTADOS POR FOLD",
        "-" * 60,
    ]

    for fold in resultados['folds']:
        lineas.append(
            f"  Fold {fold['fold']}: "
            f"Acc={fold['accuracy']:.4f}  "
            f"Prec={fold['precision']:.4f}  "
            f"Rec={fold['recall']:.4f}  "
            f"F1={fold['f1_score']:.4f}  "
            f"({fold['train_size']}/{fold['val_size']})"
        )

    lineas.extend([
        "",
        "-" * 60,
        "  MÉTRICAS PROMEDIO",
        "-" * 60,
        f"  Accuracy:  {resultados['accuracy_promedio']:.4f} ± {resultados['accuracy_std']:.4f}",
        f"  Precision: {resultados['precision_promedio']:.4f}",
        f"  Recall:    {resultados['recall_promedio']:.4f}",
        f"  F1-Score:  {resultados['f1_promedio']:.4f}",
        "",
        "-" * 60,
        "  REPORTE POR CLASE",
        "-" * 60,
    ])

    reporte = resultados['reporte_clasificacion']
    for clase in resultados['clases']:
        if clase in reporte:
            r = reporte[clase]
            lineas.append(
                f"  {clase:30s}  "
                f"Prec={r['precision']:.3f}  "
                f"Rec={r['recall']:.3f}  "
                f"F1={r['f1-score']:.3f}  "
                f"N={r['support']}"
            )

    lineas.extend(["", "=" * 60])
    return '\n'.join(lineas)
