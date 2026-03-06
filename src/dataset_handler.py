"""
dataset_handler.py
------------------
Manages a collection of labelled spectrograms that form the training dataset.
Supports in-memory accumulation, export to HDF5 / NPZ, and reload.
"""
import io
from typing import Optional

import h5py
import numpy as np
import pandas as pd


class DatasetHandler:
    """Stores labelled spectrograms and handles serialisation."""

    def __init__(self) -> None:
        self._entries: list[dict] = []

    # ------------------------------------------------------------------
    # Adding entries
    # ------------------------------------------------------------------

    def add_entry(
        self,
        spectrogram: np.ndarray,
        brand: str,
        drone_type: str,
        filename: str,
    ) -> None:
        """Append a labelled spectrogram.

        Parameters
        ----------
        spectrogram:
            2-D float array with values in ``[0, 1]``.
        brand:
            Manufacturer name, e.g. ``'DJI'``.
        drone_type:
            Model / type, e.g. ``'Phantom4'``.
        filename:
            Source file name (used for provenance).
        """
        self._entries.append(
            {
                "spectrogram": np.array(spectrogram, dtype=np.float32),
                "brand": brand,
                "type": drone_type,
                "filename": filename,
                "label": f"{brand}_{drone_type}",
            }
        )

    def clear(self) -> None:
        """Remove all entries from the in-memory dataset."""
        self._entries.clear()

    # ------------------------------------------------------------------
    # Properties / introspection
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[dict]:
        return self._entries

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    @property
    def num_classes(self) -> int:
        return len({e["label"] for e in self._entries})

    @property
    def brands(self) -> list[str]:
        seen: list[str] = []
        for e in self._entries:
            if e["brand"] not in seen:
                seen.append(e["brand"])
        return seen

    @property
    def types(self) -> list[str]:
        seen: list[str] = []
        for e in self._entries:
            if e["type"] not in seen:
                seen.append(e["type"])
        return seen

    # ------------------------------------------------------------------
    # DataFrame / array views
    # ------------------------------------------------------------------

    def get_dataframe(self) -> pd.DataFrame:
        """Return dataset metadata as a :class:`pandas.DataFrame`."""
        if not self._entries:
            return pd.DataFrame(columns=["filename", "brand", "type", "label"])
        return pd.DataFrame(
            [
                {
                    "filename": e["filename"],
                    "brand": e["brand"],
                    "type": e["type"],
                    "label": e["label"],
                }
                for e in self._entries
            ]
        )

    def get_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return ``(X, y, unique_labels)`` ready for model training.

        ``X`` shape: ``(n_samples, height, width)``
        ``y`` shape: ``(n_samples,)`` – integer class indices
        ``unique_labels``: sorted list of label strings matching indices.
        """
        if not self._entries:
            raise ValueError("Dataset is empty.")
        X = np.array([e["spectrogram"] for e in self._entries], dtype=np.float32)
        raw_labels = [e["label"] for e in self._entries]
        unique_labels = sorted(set(raw_labels))
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        y = np.array([label_to_idx[lbl] for lbl in raw_labels], dtype=np.int64)
        return X, y, unique_labels

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_hdf5(self) -> Optional[io.BytesIO]:
        """Serialise the dataset to HDF5 and return an in-memory buffer."""
        if not self._entries:
            return None
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            specs = np.array([e["spectrogram"] for e in self._entries], dtype=np.float32)
            f.create_dataset("spectrograms", data=specs, compression="gzip")
            f.create_dataset(
                "labels",
                data=np.array([e["label"].encode() for e in self._entries]),
            )
            f.create_dataset(
                "brands",
                data=np.array([e["brand"].encode() for e in self._entries]),
            )
            f.create_dataset(
                "types",
                data=np.array([e["type"].encode() for e in self._entries]),
            )
            f.create_dataset(
                "filenames",
                data=np.array([e["filename"].encode() for e in self._entries]),
            )
        buf.seek(0)
        return buf

    def export_npz(self) -> Optional[io.BytesIO]:
        """Serialise the dataset to NPZ and return an in-memory buffer."""
        if not self._entries:
            return None
        buf = io.BytesIO()
        np.savez(
            buf,
            spectrograms=np.array([e["spectrogram"] for e in self._entries], dtype=np.float32),
            labels=np.array([e["label"] for e in self._entries]),
            brands=np.array([e["brand"] for e in self._entries]),
            types=np.array([e["type"] for e in self._entries]),
            filenames=np.array([e["filename"] for e in self._entries]),
        )
        buf.seek(0)
        return buf

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def load_hdf5(self, file_obj) -> None:
        """Load entries from an HDF5 file, **replacing** current contents."""
        self._entries.clear()
        with h5py.File(io.BytesIO(file_obj.read()), "r") as f:
            specs = f["spectrograms"][:]
            labels = [v.decode() for v in f["labels"][:]]
            brands = [v.decode() for v in f["brands"][:]]
            types = [v.decode() for v in f["types"][:]]
            filenames = [v.decode() for v in f["filenames"][:]]
        for spec, lbl, brand, dtype, fname in zip(specs, labels, brands, types, filenames):
            self._entries.append(
                {
                    "spectrogram": spec.astype(np.float32),
                    "brand": brand,
                    "type": dtype,
                    "filename": fname,
                    "label": lbl,
                }
            )

    def load_npz(self, file_obj) -> None:
        """Load entries from an NPZ file, **replacing** current contents."""
        self._entries.clear()
        data = np.load(io.BytesIO(file_obj.read()), allow_pickle=False)
        for spec, lbl, brand, dtype, fname in zip(
            data["spectrograms"],
            data["labels"],
            data["brands"],
            data["types"],
            data["filenames"],
        ):
            self._entries.append(
                {
                    "spectrogram": spec.astype(np.float32),
                    "brand": str(brand),
                    "type": str(dtype),
                    "filename": str(fname),
                    "label": str(lbl),
                }
            )
