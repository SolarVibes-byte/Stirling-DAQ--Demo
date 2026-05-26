# Kinematics.py — volume models (sinusoidal, crank-slider, Fourier)
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import Configuration as Cfg
from Configuration import TDCReference

def _apply_tdc_convention(theta_rad, tdc_ref):
    if tdc_ref == TDCReference.PISTON:
        return theta_rad
    elif tdc_ref == TDCReference.DISPLACER:
        return theta_rad - Cfg.PHASE_OFFSET_RAD
    raise ValueError(f"Unknown TDC reference: {tdc_ref}")

def _crank_slider_position(theta_rad, crank_radius, rod_length):
    lam = rod_length / crank_radius
    sin_t = np.sin(theta_rad); cos_t = np.cos(theta_rad)
    return 0.5 * (1.0 - cos_t + lam - np.sqrt(lam**2 - sin_t**2))

def _crank_slider_velocity(theta_rad, crank_radius, rod_length):
    lam = rod_length / crank_radius
    sin_t = np.sin(theta_rad); cos_t = np.cos(theta_rad)
    return 0.5 * sin_t * (1.0 + cos_t / np.sqrt(lam**2 - sin_t**2))

def volume_sinusoidal(theta_rad, head=1, porosity_percent=62,
                      tdc_ref=Cfg.TDC_REFERENCE_IDEAL):
    theta = _apply_tdc_convention(theta_rad, tdc_ref)
    pos_piston = 0.5 * (1.0 - np.cos(theta))
    pos_displacer = 0.5 * (1.0 - np.cos(theta - Cfg.PHASE_OFFSET_RAD))
    V_c = Cfg.V_SWEPT_COMPRESSION_M3 * (1.0 - pos_piston) + Cfg.V_DEAD_COOLER_M3
    V_e = Cfg.V_SWEPT_EXPANSION_M3 * pos_displacer + Cfg.get_heater_volume(head)
    V_fixed = Cfg.get_total_fixed_volume(head, porosity_percent)
    V_total = V_c + V_e + (V_fixed - Cfg.V_DEAD_COOLER_M3 - Cfg.get_heater_volume(head))
    return V_c, V_e, V_total

def volume_crank_slider(theta_rad, head=1, porosity_percent=62,
                        tdc_ref=Cfg.TDC_REFERENCE_IDEAL):
    theta = _apply_tdc_convention(theta_rad, tdc_ref)
    pos_piston = _crank_slider_position(theta, Cfg.PISTON_CRANK_RADIUS_M, Cfg.PISTON_ROD_LENGTH_M)
    pos_displacer = _crank_slider_position(theta - Cfg.PHASE_OFFSET_RAD,
                                           Cfg.DISPLACER_CRANK_RADIUS_M, Cfg.DISPLACER_ROD_LENGTH_M)
    V_c = Cfg.V_SWEPT_COMPRESSION_M3 * (1.0 - pos_piston) + Cfg.V_DEAD_COOLER_M3
    V_e = Cfg.V_SWEPT_EXPANSION_M3 * pos_displacer + Cfg.get_heater_volume(head)
    V_fixed = Cfg.get_total_fixed_volume(head, porosity_percent)
    V_total = V_c + V_e + (V_fixed - Cfg.V_DEAD_COOLER_M3 - Cfg.get_heater_volume(head))
    return V_c, V_e, V_total

def volume_derivatives_crank_slider(theta_rad, tdc_ref=Cfg.TDC_REFERENCE_IDEAL):
    theta = _apply_tdc_convention(theta_rad, tdc_ref)
    dpos_piston = _crank_slider_velocity(theta, Cfg.PISTON_CRANK_RADIUS_M, Cfg.PISTON_ROD_LENGTH_M)
    dpos_displacer = _crank_slider_velocity(theta - Cfg.PHASE_OFFSET_RAD,
                                            Cfg.DISPLACER_CRANK_RADIUS_M, Cfg.DISPLACER_ROD_LENGTH_M)
    dV_c = -Cfg.V_SWEPT_COMPRESSION_M3 * dpos_piston
    dV_e = Cfg.V_SWEPT_EXPANSION_M3 * dpos_displacer
    return dV_c, dV_e

# Truncated-Fourier volume of the engine, provided by N. Lummen for the ISEC
# article (private communication). The crank angle is the displacer angle in
# radians, with the displacer piston at top dead centre when alpha = 0.
# Each swept space is a cosine series (amplitude_m3, harmonic_order, phase_rad);
# the displacer-side series is the compression space and the piston-side series
# is the expansion space, with the remaining dead volume lumped in the baseline.
_FOURIER_BASELINE_M3: float = 1.90e-4

_FOURIER_COMPRESSION: list[tuple[float, int, float]] = [
    (6.552e-5, 0,  0.000),
    (6.096e-5, 1,  1.045),
    (4.520e-6, 2,  0.323),
    (1.000e-8, 3, -3.051),
]
_FOURIER_EXPANSION: list[tuple[float, int, float]] = [
    (7.357e-5, 0,  0.000),
    (6.321e-5, 1, -3.142),
    (3.580e-6, 2, -3.142),
    (1.000e-8, 3,  0.000),
]


def _fourier_series(theta_rad: NDArray[np.float64],
                   coeffs: list[tuple[float, int, float]]) -> NDArray[np.float64]:
    """Evaluate a truncated cosine series sum(a_n cos(n theta + phi_n))."""
    out = np.zeros_like(theta_rad)
    for amplitude, n, phase in coeffs:
        out = out + amplitude * np.cos(n * theta_rad + phase)
    return out


def _fourier_series_derivative(theta_rad: NDArray[np.float64],
                              coeffs: list[tuple[float, int, float]]) -> NDArray[np.float64]:
    """Analytical derivative d/d(theta) of a truncated cosine series."""
    out = np.zeros_like(theta_rad)
    for amplitude, n, phase in coeffs:
        out = out - amplitude * n * np.sin(n * theta_rad + phase)
    return out


def volume_fourier(theta_rad: NDArray[np.float64],
                  head: int = 1,
                  porosity_percent: int = 62,
                  ) -> tuple[NDArray[np.float64],
                             NDArray[np.float64],
                             NDArray[np.float64]]:
    """Sage truncated-Fourier volumes: compression, expansion, and total.

    The compression and expansion swept volumes follow the supervisor's cosine
    series; the lumped dead volume is added through the baseline. This is the
    reference volume model of the work. The crank angle is used as supplied
    (displacer TDC at alpha = 0), with no TDC transformation, matching the
    convention of the measured data. The ``head`` and ``porosity_percent``
    arguments are accepted for interface symmetry with the crank-slider model.

    Returns
    -------
    (V_c, V_e, V_total) in m^3, each as an array of shape (N,).
    """
    V_c = _fourier_series(theta_rad, _FOURIER_COMPRESSION)
    V_e = _fourier_series(theta_rad, _FOURIER_EXPANSION)
    V_total = _FOURIER_BASELINE_M3 + V_c + V_e
    return V_c, V_e, V_total


def volume_derivatives_fourier(theta_rad: NDArray[np.float64],
                             ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Analytical dV_c/d(theta) and dV_e/d(theta) for the Fourier model.

    Used by the Adiabatic ODE integrator so that the second-order model relies
    on the same Sage volume representation as the rest of the pipeline.

    Returns
    -------
    (dV_c, dV_e) in m^3/rad, each as an array of shape (N,).
    """
    dV_c = _fourier_series_derivative(theta_rad, _FOURIER_COMPRESSION)
    dV_e = _fourier_series_derivative(theta_rad, _FOURIER_EXPANSION)
    return dV_c, dV_e
