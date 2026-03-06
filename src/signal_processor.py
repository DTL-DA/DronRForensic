"""
signal_processor.py
-------------------
Handles loading RF signal files in various formats and converting them to
STFT-based spectrograms ready for CNN classification.
"""
import io

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import zoom


class SignalProcessor:
    """Loads RF signals and computes STFT spectrograms."""

    SUPPORTED_FORMATS = ("csv", "npy", "bin", "dat", "wav")

    def __init__(
        self,
        fs: float = 2e6,
        nperseg: int = 256,
        noverlap: int | None = None,
        nfft: int = 512,
        window: str = "hann",
    ) -> None:
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg * 3 // 4
        self.nfft = nfft
        self.window = window

    # ------------------------------------------------------------------
    # Signal loading
    # ------------------------------------------------------------------

    def load_signal(self, file_obj, file_type: str) -> np.ndarray:
        """Load an RF signal from a file-like object.

        Supported *file_type* values:
        - ``csv``: columns **I** and **Q** (or first two columns).
        - ``npy``: NumPy array (complex or two-column real).
        - ``bin`` / ``dat``: interleaved float32 I/Q binary stream.
        - ``wav``: WAV file (mono → amplitude; stereo → I/Q).

        Returns a 1-D complex NumPy array.
        """
        file_type = file_type.lower()
        if file_type == "dat":
            file_type = "bin"

        if file_type == "csv":
            return self._load_csv(file_obj)
        elif file_type == "npy":
            return self._load_npy(file_obj)
        elif file_type == "bin":
            return self._load_bin(file_obj)
        elif file_type == "wav":
            return self._load_wav(file_obj)
        else:
            raise ValueError(
                f"Unsupported file type '{file_type}'. "
                f"Choose from: {self.SUPPORTED_FORMATS}"
            )

    def _load_csv(self, file_obj) -> np.ndarray:
        df = pd.read_csv(file_obj)
        cols_upper = {c.upper(): c for c in df.columns}
        if "I" in cols_upper and "Q" in cols_upper:
            return (
                df[cols_upper["I"]].values + 1j * df[cols_upper["Q"]].values
            )
        elif df.shape[1] >= 2:
            return df.iloc[:, 0].values + 1j * df.iloc[:, 1].values
        else:
            return df.iloc[:, 0].values.astype(complex)

    def _load_npy(self, file_obj) -> np.ndarray:
        data = np.load(io.BytesIO(file_obj.read()), allow_pickle=False)
        if np.iscomplexobj(data):
            return data.flatten()
        if data.ndim == 2 and data.shape[1] == 2:
            return data[:, 0] + 1j * data[:, 1]
        return data.flatten().astype(complex)

    def _load_bin(self, file_obj) -> np.ndarray:
        raw = np.frombuffer(file_obj.read(), dtype=np.float32)
        if len(raw) >= 2 and len(raw) % 2 == 0:
            return raw[::2] + 1j * raw[1::2]
        return raw.astype(complex)

    def _load_wav(self, file_obj) -> np.ndarray:
        sample_rate, data = wavfile.read(io.BytesIO(file_obj.read()))
        self.fs = float(sample_rate)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 0] + 1j * data[:, 1]
        return data.flatten().astype(complex)

    # ------------------------------------------------------------------
    # STFT & spectrogram
    # ------------------------------------------------------------------

    def compute_stft(
        self, signal_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the STFT and return ``(f, t, spectrogram_dB)``.

        The spectrogram is the magnitude of the STFT expressed in dB.
        """
        is_complex = np.iscomplexobj(signal_data)
        f, t, Zxx = signal.stft(
            signal_data,
            fs=self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            return_onesided=not is_complex,
        )
        magnitude = np.abs(Zxx)
        magnitude = np.maximum(magnitude, 1e-10)
        spectrogram_db = 20.0 * np.log10(magnitude)
        return f, t, spectrogram_db

    @staticmethod
    def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
        """Normalize spectrogram values to ``[0, 1]``."""
        s_min, s_max = spec.min(), spec.max()
        if s_max - s_min > 0:
            return (spec - s_min) / (s_max - s_min)
        return np.zeros_like(spec)

    @staticmethod
    def resize_spectrogram(
        spec: np.ndarray, target_size: tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """Resize spectrogram to *target_size* using bilinear interpolation."""
        zoom_factors = (
            target_size[0] / spec.shape[0],
            target_size[1] / spec.shape[1],
        )
        return zoom(spec, zoom_factors, order=1)

    def process_signal(
        self,
        signal_data: np.ndarray,
        target_size: tuple[int, int] = (128, 128),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Full pipeline: STFT → magnitude (dB) → normalize → resize.

        Returns ``(f, t, spectrogram_dB, spectrogram_normalized)``.
        """
        f, t, spec_db = self.compute_stft(signal_data)
        spec_norm = self.normalize_spectrogram(spec_db)
        if spec_norm.shape != target_size:
            spec_resized = self.resize_spectrogram(spec_norm, target_size)
        else:
            spec_resized = spec_norm
        return f, t, spec_db, spec_resized

    # ------------------------------------------------------------------
    # Synthetic signal generation (for testing / demos)
    # ------------------------------------------------------------------

    def generate_synthetic_signal(
        self,
        drone_type: str = "quadcopter",
        duration: float = 0.1,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """Generate a synthetic drone-like RF signal for demos and tests.

        Parameters
        ----------
        drone_type:
            One of ``'quadcopter'``, ``'fixed_wing'``, or ``'hexacopter'``.
        duration:
            Signal duration in seconds.
        noise_level:
            Standard deviation of additive complex Gaussian noise.
        """
        n_samples = int(self.fs * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        if drone_type == "quadcopter":
            hop_freq = 1e6
            sig = np.exp(1j * 2 * np.pi * (hop_freq * t + 0.5 * hop_freq * t ** 2))
            sig += 0.3 * np.exp(1j * 2 * np.pi * 2 * hop_freq * t)
        elif drone_type == "fixed_wing":
            hop_freq = 2e6
            sig = np.exp(1j * 2 * np.pi * hop_freq * np.sin(2 * np.pi * 5 * t))
        elif drone_type == "hexacopter":
            hop_freq = 1.5e6
            sig = np.exp(1j * 2 * np.pi * hop_freq * t)
            sig += 0.5 * np.exp(1j * 2 * np.pi * 3 * hop_freq * t * np.cos(2 * np.pi * 3 * t))
        else:
            sig = np.exp(1j * 2 * np.pi * 1e6 * t)

        rng = np.random.default_rng(seed=hash(drone_type) & 0xFFFFFFFF)
        noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) * noise_level
        return sig + noise
