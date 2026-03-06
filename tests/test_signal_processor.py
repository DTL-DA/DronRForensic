"""
Tests for src/signal_processor.py
"""
import io

import numpy as np
import pandas as pd
import pytest

from src.signal_processor import SignalProcessor


@pytest.fixture
def processor():
    return SignalProcessor(fs=2e6, nperseg=64, noverlap=48, nfft=128, window="hann")


# ---------------------------------------------------------------------------
# Synthetic signal generation
# ---------------------------------------------------------------------------

class TestSyntheticSignal:
    def test_quadcopter_length(self, processor):
        sig = processor.generate_synthetic_signal("quadcopter", duration=0.01)
        expected = int(processor.fs * 0.01)
        assert len(sig) == expected

    def test_fixed_wing_length(self, processor):
        sig = processor.generate_synthetic_signal("fixed_wing", duration=0.02)
        assert len(sig) == int(processor.fs * 0.02)

    def test_hexacopter_length(self, processor):
        sig = processor.generate_synthetic_signal("hexacopter", duration=0.005)
        assert len(sig) == int(processor.fs * 0.005)

    def test_output_is_complex(self, processor):
        sig = processor.generate_synthetic_signal()
        assert np.iscomplexobj(sig)

    def test_noise_level_zero_no_randomness(self, processor):
        s1 = processor.generate_synthetic_signal("quadcopter", noise_level=0.0)
        s2 = processor.generate_synthetic_signal("quadcopter", noise_level=0.0)
        np.testing.assert_array_equal(s1, s2)

    def test_unknown_type_fallback(self, processor):
        sig = processor.generate_synthetic_signal("unknown_type", duration=0.01)
        assert len(sig) == int(processor.fs * 0.01)


# ---------------------------------------------------------------------------
# STFT computation
# ---------------------------------------------------------------------------

class TestComputeSTFT:
    def test_output_shapes(self, processor):
        sig = processor.generate_synthetic_signal(duration=0.01)
        f, t, spec = processor.compute_stft(sig)
        # spec should be 2-D
        assert spec.ndim == 2
        assert f.ndim == 1
        assert t.ndim == 1

    def test_spec_finite(self, processor):
        sig = processor.generate_synthetic_signal(duration=0.01)
        _, _, spec = processor.compute_stft(sig)
        assert np.all(np.isfinite(spec))

    def test_spec_in_db_range(self, processor):
        sig = processor.generate_synthetic_signal(duration=0.01)
        _, _, spec = processor.compute_stft(sig)
        # dB values should be finite and typically > -200
        assert spec.min() > -300


# ---------------------------------------------------------------------------
# Normalisation & resizing
# ---------------------------------------------------------------------------

class TestNormalizeSpectrogram:
    def test_range(self, processor):
        sig = processor.generate_synthetic_signal(duration=0.01)
        _, _, spec = processor.compute_stft(sig)
        norm = processor.normalize_spectrogram(spec)
        assert norm.min() >= 0.0 - 1e-9
        assert norm.max() <= 1.0 + 1e-9

    def test_constant_spec_returns_zeros(self):
        spec = np.full((10, 10), 5.0)
        norm = SignalProcessor.normalize_spectrogram(spec)
        np.testing.assert_array_equal(norm, np.zeros((10, 10)))

    def test_shape_preserved(self, processor):
        sig = processor.generate_synthetic_signal(duration=0.01)
        _, _, spec = processor.compute_stft(sig)
        norm = processor.normalize_spectrogram(spec)
        assert norm.shape == spec.shape


class TestResizeSpectrogram:
    def test_resize_larger(self):
        spec = np.random.rand(32, 16)
        resized = SignalProcessor.resize_spectrogram(spec, (64, 64))
        assert resized.shape == (64, 64)

    def test_resize_smaller(self):
        spec = np.random.rand(256, 128)
        resized = SignalProcessor.resize_spectrogram(spec, (64, 32))
        assert resized.shape == (64, 32)

    def test_no_resize_needed(self):
        spec = np.random.rand(128, 128)
        resized = SignalProcessor.resize_spectrogram(spec, (128, 128))
        assert resized.shape == (128, 128)


# ---------------------------------------------------------------------------
# Full process_signal pipeline
# ---------------------------------------------------------------------------

class TestProcessSignal:
    def test_output_target_size(self, processor):
        sig = processor.generate_synthetic_signal(duration=0.01)
        f, t, spec_db, spec_norm = processor.process_signal(sig, (64, 64))
        assert spec_norm.shape == (64, 64)

    def test_norm_range(self, processor):
        sig = processor.generate_synthetic_signal(duration=0.01)
        _, _, _, spec_norm = processor.process_signal(sig, (64, 64))
        assert spec_norm.min() >= 0.0 - 1e-9
        assert spec_norm.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Signal loading from file-like objects
# ---------------------------------------------------------------------------

class TestLoadSignal:
    def _make_iq_array(self, n: int = 200) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def test_load_csv_iq_columns(self, processor):
        iq = self._make_iq_array()
        df = pd.DataFrame({"I": iq.real, "Q": iq.imag})
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        loaded = processor.load_signal(buf, "csv")
        np.testing.assert_allclose(loaded.real, iq.real, atol=1e-5)
        np.testing.assert_allclose(loaded.imag, iq.imag, atol=1e-5)

    def test_load_csv_generic_columns(self, processor):
        iq = self._make_iq_array()
        df = pd.DataFrame({"col_a": iq.real, "col_b": iq.imag})
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        loaded = processor.load_signal(buf, "csv")
        assert len(loaded) == len(iq)

    def test_load_csv_single_column(self, processor):
        data = np.arange(100, dtype=float)
        df = pd.DataFrame({"amp": data})
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        loaded = processor.load_signal(buf, "csv")
        assert len(loaded) == 100

    def test_load_npy_complex(self, processor):
        iq = self._make_iq_array(128)
        buf = io.BytesIO()
        np.save(buf, iq)
        buf.seek(0)
        loaded = processor.load_signal(buf, "npy")
        np.testing.assert_allclose(loaded, iq, atol=1e-6)

    def test_load_npy_two_column_real(self, processor):
        iq = self._make_iq_array(64)
        arr = np.stack([iq.real, iq.imag], axis=1)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        loaded = processor.load_signal(buf, "npy")
        np.testing.assert_allclose(loaded.real, iq.real, atol=1e-5)
        np.testing.assert_allclose(loaded.imag, iq.imag, atol=1e-5)

    def test_load_bin(self, processor):
        iq = self._make_iq_array(50)
        interleaved = np.empty(100, dtype=np.float32)
        interleaved[::2] = iq.real
        interleaved[1::2] = iq.imag
        buf = io.BytesIO(interleaved.tobytes())
        loaded = processor.load_signal(buf, "bin")
        np.testing.assert_allclose(loaded.real, iq.real, atol=1e-5)
        np.testing.assert_allclose(loaded.imag, iq.imag, atol=1e-5)

    def test_load_dat_alias(self, processor):
        """'.dat' should be treated the same as '.bin'."""
        iq = self._make_iq_array(50)
        interleaved = np.empty(100, dtype=np.float32)
        interleaved[::2] = iq.real
        interleaved[1::2] = iq.imag
        buf = io.BytesIO(interleaved.tobytes())
        loaded = processor.load_signal(buf, "dat")
        np.testing.assert_allclose(loaded.real, iq.real, atol=1e-5)

    def test_unsupported_format_raises(self, processor):
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.load_signal(io.BytesIO(b""), "xyz")
