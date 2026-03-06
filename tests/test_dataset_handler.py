"""
Tests for src/dataset_handler.py
"""
import io

import numpy as np
import pytest

from src.dataset_handler import DatasetHandler


def _make_spec(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((size, size), dtype=np.float32)


@pytest.fixture
def handler():
    return DatasetHandler()


@pytest.fixture
def populated_handler():
    h = DatasetHandler()
    h.add_entry(_make_spec(0), "DJI", "Phantom4", "file1.npy")
    h.add_entry(_make_spec(1), "DJI", "Mavic2", "file2.npy")
    h.add_entry(_make_spec(2), "Parrot", "Bebop2", "file3.npy")
    h.add_entry(_make_spec(3), "Parrot", "ANAFI", "file4.npy")
    return h


# ---------------------------------------------------------------------------
# add_entry / properties
# ---------------------------------------------------------------------------

class TestAddEntry:
    def test_num_entries_increments(self, handler):
        assert handler.num_entries == 0
        handler.add_entry(_make_spec(), "DJI", "Phantom4", "f.npy")
        assert handler.num_entries == 1

    def test_num_classes(self, populated_handler):
        assert populated_handler.num_classes == 4

    def test_label_format(self, handler):
        handler.add_entry(_make_spec(), "DJI", "Phantom4", "f.npy")
        assert handler.entries[0]["label"] == "DJI_Phantom4"

    def test_brands_property(self, populated_handler):
        brands = populated_handler.brands
        assert "DJI" in brands
        assert "Parrot" in brands

    def test_types_property(self, populated_handler):
        types = populated_handler.types
        assert "Phantom4" in types
        assert "Bebop2" in types


# ---------------------------------------------------------------------------
# get_dataframe
# ---------------------------------------------------------------------------

class TestGetDataframe:
    def test_empty_returns_empty_df(self, handler):
        df = handler.get_dataframe()
        assert len(df) == 0
        assert list(df.columns) == ["filename", "brand", "type", "label"]

    def test_columns(self, populated_handler):
        df = populated_handler.get_dataframe()
        assert set(df.columns) == {"filename", "brand", "type", "label"}

    def test_row_count(self, populated_handler):
        df = populated_handler.get_dataframe()
        assert len(df) == 4


# ---------------------------------------------------------------------------
# get_arrays
# ---------------------------------------------------------------------------

class TestGetArrays:
    def test_empty_raises(self, handler):
        with pytest.raises(ValueError, match="empty"):
            handler.get_arrays()

    def test_shapes(self, populated_handler):
        X, y, labels = populated_handler.get_arrays()
        assert X.shape == (4, 64, 64)
        assert y.shape == (4,)
        assert len(labels) == 4

    def test_labels_sorted(self, populated_handler):
        _, _, labels = populated_handler.get_arrays()
        assert labels == sorted(labels)

    def test_y_values_in_range(self, populated_handler):
        X, y, labels = populated_handler.get_arrays()
        assert y.min() >= 0
        assert y.max() < len(labels)

    def test_x_dtype(self, populated_handler):
        X, _, _ = populated_handler.get_arrays()
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# HDF5 export / import round-trip
# ---------------------------------------------------------------------------

class TestHDF5RoundTrip:
    def test_export_returns_bytes_io(self, populated_handler):
        buf = populated_handler.export_hdf5()
        assert buf is not None
        assert buf.read(4) == b"\x89HDF"  # HDF5 magic bytes

    def test_empty_export_returns_none(self, handler):
        assert handler.export_hdf5() is None

    def test_round_trip(self, populated_handler):
        buf = populated_handler.export_hdf5()

        h2 = DatasetHandler()
        h2.load_hdf5(buf)

        assert h2.num_entries == populated_handler.num_entries
        assert h2.num_classes == populated_handler.num_classes

        # Spectrogram content preserved
        for orig, loaded in zip(populated_handler.entries, h2.entries):
            np.testing.assert_allclose(orig["spectrogram"], loaded["spectrogram"], atol=1e-5)
            assert orig["brand"] == loaded["brand"]
            assert orig["type"] == loaded["type"]
            assert orig["label"] == loaded["label"]

    def test_load_replaces_existing(self, populated_handler):
        buf = populated_handler.export_hdf5()

        h2 = DatasetHandler()
        h2.add_entry(_make_spec(99), "Autel", "EVO", "extra.npy")
        assert h2.num_entries == 1

        h2.load_hdf5(buf)
        assert h2.num_entries == populated_handler.num_entries  # old entry replaced


# ---------------------------------------------------------------------------
# NPZ export / import round-trip
# ---------------------------------------------------------------------------

class TestNPZRoundTrip:
    def test_export_returns_bytes_io(self, populated_handler):
        buf = populated_handler.export_npz()
        assert buf is not None
        assert isinstance(buf, io.BytesIO)

    def test_empty_export_returns_none(self, handler):
        assert handler.export_npz() is None

    def test_round_trip(self, populated_handler):
        buf = populated_handler.export_npz()

        h2 = DatasetHandler()
        h2.load_npz(buf)

        assert h2.num_entries == populated_handler.num_entries
        assert h2.num_classes == populated_handler.num_classes

        for orig, loaded in zip(populated_handler.entries, h2.entries):
            np.testing.assert_allclose(orig["spectrogram"], loaded["spectrogram"], atol=1e-5)
            assert orig["brand"] == loaded["brand"]
            assert orig["label"] == loaded["label"]


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_dataset(self, populated_handler):
        populated_handler.clear()
        assert populated_handler.num_entries == 0
        assert populated_handler.num_classes == 0
