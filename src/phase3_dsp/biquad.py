"""
Phase 3: Robert Bristow-Johnson (RBJ) Audio EQ Biquad Filter Mathematics
MA375TGR | Mathematics of Music
Implements deterministic biquad filter coefficient calculation and transfer functions.
"""

from typing import Tuple
import numpy as np


class BiquadFilter:
    """
    Second-Order IIR (Biquad) Filter using the RBJ Audio EQ Cookbook equations.
    Transfer function:
            b0 + b1*z^-1 + b2*z^-2
    H(z) = ------------------------
            a0 + a1*z^-1 + a2*z^-2
    """

    def __init__(
        self,
        filter_type: str,
        f0_hz: float,
        gain_db: float = 0.0,
        q: float = 1.0,
        fs: int = 44100
    ):
        self.filter_type = filter_type.lower()
        self.f0 = float(f0_hz)
        self.gain_db = float(gain_db)
        self.q = max(0.05, float(q))
        self.fs = int(fs)

        # Compute coefficients
        self.b, self.a = self._compute_coefficients()

    def _compute_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates normalized biquad coefficients [b0, b1, b2] and [a0=1, a1, a2]."""
        # Intermediate variables
        A = 10.0 ** (self.gain_db / 40.0)
        w0 = 2.0 * np.pi * self.f0 / self.fs
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * self.q)

        # Default flat coefficients
        b0, b1, b2 = 1.0, 0.0, 0.0
        a0, a1, a2 = 1.0, 0.0, 0.0

        if self.filter_type == "peaking":
            # Peaking EQ (Bell) per Slide 12
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A

        elif self.filter_type == "lowshelf":
            # Low-Shelf EQ
            beta = 2.0 * np.sqrt(A) * alpha
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + beta)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - beta)
            a0 = (A + 1.0) + (A - 1.0) * cos_w0 + beta
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
            a2 = (A + 1.0) + (A - 1.0) * cos_w0 - beta

        elif self.filter_type == "highshelf":
            # High-Shelf EQ
            beta = 2.0 * np.sqrt(A) * alpha
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + beta)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - beta)
            a0 = (A + 1.0) - (A - 1.0) * cos_w0 + beta
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
            a2 = (A + 1.0) - (A - 1.0) * cos_w0 - beta

        elif self.filter_type == "highpass":
            # High-Pass filter (Rumble cut)
            b0 = (1.0 + cos_w0) / 2.0
            b1 = -(1.0 + cos_w0)
            b2 = (1.0 + cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif self.filter_type == "notch":
            # Notch filter
            b0 = 1.0
            b1 = -2.0 * cos_w0
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        # Normalize by a0
        b_norm = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a_norm = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

        return b_norm, a_norm

    def get_sos(self) -> np.ndarray:
        """Returns standard Second-Order Section (SOS) format for scipy."""
        return np.array([[self.b[0], self.b[1], self.b[2], 1.0, self.a[1], self.a[2]]], dtype=np.float64)

    def frequency_response(self, freqs_hz: np.ndarray) -> np.ndarray:
        """
        Calculates complex frequency response H(e^jw) at given frequencies:
        H(w) = (b0 + b1*e^-jw + b2*e^-j2w) / (1 + a1*e^-jw + a2*e^-j2w)
        """
        w = 2.0 * np.pi * freqs_hz / self.fs
        z_1 = np.exp(-1j * w)
        z_2 = np.exp(-2j * w)

        num = self.b[0] + self.b[1] * z_1 + self.b[2] * z_2
        den = 1.0 + self.a[1] * z_1 + self.a[2] * z_2

        return num / den

    def magnitude_response_db(self, freqs_hz: np.ndarray) -> np.ndarray:
        """Returns magnitude response in dB at the specified frequencies."""
        h = self.frequency_response(freqs_hz)
        return 20.0 * np.log10(np.abs(h) + 1e-12)

