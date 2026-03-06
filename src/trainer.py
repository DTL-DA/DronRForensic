"""
trainer.py
----------
Stratified K-fold cross-validation and final model training utilities.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


class ModelTrainer:
    """Orchestrates cross-validation and final model training.

    Parameters
    ----------
    n_splits:
        Number of folds for stratified K-fold cross-validation.
    epochs:
        Maximum number of training epochs per fold / final run.
    batch_size:
        Mini-batch size.
    """

    def __init__(
        self,
        n_splits: int = 5,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.cv_results: list[dict] = []
        self.final_history = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_channel(X: np.ndarray) -> np.ndarray:
        """Add a trailing channel dimension: ``(N, H, W) → (N, H, W, 1)``."""
        return X[..., np.newaxis]

    def _get_callbacks(self, patience: int = 10):
        import tensorflow as tf

        return [
            tf.keras.callbacks.EarlyStopping(
                patience=patience, restore_best_weights=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=max(3, patience // 2), verbose=0
            ),
        ]

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        model_builder: Callable,
        X: np.ndarray,
        y: np.ndarray,
        labels: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Stratified K-fold cross-validation.

        Parameters
        ----------
        model_builder:
            Zero-argument callable that returns a *compiled* Keras model.
        X:
            Spectrogram array of shape ``(n_samples, H, W)``.
        y:
            Integer class labels of shape ``(n_samples,)``.
        labels:
            Ordered list of class-name strings (index → name mapping).
        progress_callback:
            Optional ``(fold, total)`` callback for UI progress updates.

        Returns
        -------
        list of dict
            One entry per fold containing metrics and history.
        """
        X_c = self._add_channel(X)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.cv_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_c, y), start=1):
            if progress_callback is not None:
                progress_callback(fold, self.n_splits)

            X_tr, X_val = X_c[train_idx], X_c[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = model_builder()
            history = model.fit(
                X_tr,
                y_tr,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=self._get_callbacks(patience=10),
                verbose=0,
            )

            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            report = classification_report(
                y_val, y_pred, target_names=labels, output_dict=True, zero_division=0
            )

            self.cv_results.append(
                {
                    "fold": fold,
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "y_val": y_val,
                    "y_pred": y_pred,
                    "report": report,
                    "history": history.history,
                    "confusion_matrix": confusion_matrix(y_val, y_pred),
                }
            )

        return self.cv_results

    def get_cv_summary(self) -> dict | None:
        """Aggregate statistics across all folds."""
        if not self.cv_results:
            return None
        accs = [r["val_accuracy"] for r in self.cv_results]
        losses = [r["val_loss"] for r in self.cv_results]
        return {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "fold_accuracies": accs,
            "fold_losses": losses,
        }

    def get_cv_dataframe(self) -> pd.DataFrame:
        """Return per-fold results as a :class:`~pandas.DataFrame`."""
        if not self.cv_results:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "Fold": [r["fold"] for r in self.cv_results],
                "Accuracy": [r["val_accuracy"] for r in self.cv_results],
                "Loss": [r["val_loss"] for r in self.cv_results],
            }
        )

    # ------------------------------------------------------------------
    # Final model training
    # ------------------------------------------------------------------

    def train_final_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.1,
    ):
        """Train *model* on the complete dataset.

        Parameters
        ----------
        model:
            Compiled Keras model.
        X:
            Spectrogram array ``(n_samples, H, W)``.
        y:
            Integer class labels ``(n_samples,)``.
        val_split:
            Fraction of data to hold out for validation during training.

        Returns
        -------
        keras.callbacks.History
        """
        X_c = self._add_channel(X)
        self.final_history = model.fit(
            X_c,
            y,
            validation_split=val_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self._get_callbacks(patience=15),
            verbose=0,
        )
        return self.final_history
