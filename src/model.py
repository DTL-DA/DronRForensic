"""
model.py
--------
CNN architecture for drone RF fingerprint classification.

The model treats each normalised STFT spectrogram as a grayscale image and
learns discriminative frequency-time features using three convolutional blocks
followed by global average pooling and a dense classifier head.
"""
import io
import os
import tempfile

import numpy as np


def build_cnn(input_shape: tuple[int, int, int], num_classes: int):
    """Build and compile the drone-classification CNN.

    Parameters
    ----------
    input_shape:
        ``(height, width, channels)`` – e.g. ``(128, 128, 1)``.
    num_classes:
        Number of output classes.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        Input,
        MaxPooling2D,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    inputs = Input(shape=input_shape, name="spectrogram_input")

    # Block 1
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Classifier head
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax", name="class_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="DronRForensic_CNN")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model_to_bytes(model) -> bytes:
    """Serialise a Keras model to an in-memory ``.h5`` byte string."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save(tmp_path)
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_model_from_bytes(data: bytes):
    """Deserialise a Keras model from an ``.h5`` byte string."""
    from tensorflow.keras.models import load_model

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return load_model(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
