# ThermodynamicsAdiabatic.py
# Second-order ideal adiabatic Stirling model (Urieli & Berchowitz, 1984).
#
# Five cells: compression (c), cooler (k), regenerator (r), heater (h), expansion (e).
#   - Compression and expansion: adiabatic working spaces (T variable).
#   - Cooler and heater: isothermal HX at fixed wall temperatures.
#   - Regenerator: ideal, at log-mean temperature.
#
# Seven ODEs integrated with custom RK4 + cyclic convergence.
# Two auxiliary upwind variables (TCK, THE) carry over between RHS evaluations.
#
# The working-space volumes can be supplied by either the exact crank-slider
# kinematics or the truncated-Fourier (Sage) representation, selected through
# the volume_model argument. This keeps the second-order model on the same
# volume basis as the rest of the pipeline when the Fourier reference is used.
#
# Reference: Urieli, I. & Berchowitz, D. (1984). Stirling Cycle Engine Analysis.
# MATLAB reference: sea/adiab/{adiab.m, dadiab.m, rk4.m}.

from __future__ import annotations
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray

import Configuration as Cfg
import Kinematics as Kin
from Configuration import TDCReference


# ──────────────────────────────────────────────────────────────────────
#  1. RETURN TYPE
# ──────────────────────────────────────────────────────────────────────

class AdiabaticResult(TypedDict):
    """Diagnostic-rich output of the Adiabatic model."""
    P:                  NDArray[np.float64]   # instantaneous pressure (Pa)
    T_c:                NDArray[np.float64]   # compression-space gas T (K)
    T_e:                NDArray[np.float64]   # expansion-space gas T (K)
    n_iterations:       int                   # cycles needed to converge
    closure_error_K:    float                 # final |dTc| + |dTe| at boundary
    converged:          bool                  # True if tolerance reached


# ──────────────────────────────────────────────────────────────────────
#  2. STATE VECTOR INDEXING
# ──────────────────────────────────────────────────────────────────────

# Integrated state variables (7 ODEs)
_TC, _TE, _QK, _QR, _QH, _WC, _WE = range(7)

# Auxiliary upwind interface temperatures (carried over between RK4 steps)
_TCK, _THE = 7, 8

_STATE_SIZE = 9     # 7 ODEs + 2 carry-over


# ──────────────────────────────────────────────────────────────────────
#  3. VOLUME-MODEL DISPATCH
# ──────────────────────────────────────────────────────────────────────

def _volumes(theta_array: NDArray[np.float64],
             ctx: dict,
             ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                        NDArray[np.float64], NDArray[np.float64]]:
    """Return (V_c, V_e, dV_c, dV_e) from the configured volume model.

    The adiabatic integrator needs both the working-space volumes and their
    angular derivatives at each crank angle. This helper routes those calls to
    either the Fourier (Sage) representation or the exact crank-slider model,
    so the ODE system is agnostic to which kinematic basis is in use.

    Parameters
    ----------
    theta_array : crank angles (rad), shape (N,).
    ctx : context dict carrying "volume_model", "head", "porosity", "tdc_ref".

    Returns
    -------
    (V_c, V_e, dV_c, dV_e) with volumes in m^3 and derivatives in m^3/rad.
    """
    if ctx["volume_model"] == "fourier":
        V_c, V_e, _ = Kin.volume_fourier(theta_array, ctx["head"], ctx["porosity"])
        dV_c, dV_e = Kin.volume_derivatives_fourier(theta_array)
        return V_c, V_e, dV_c, dV_e

    V_c, V_e, _ = Kin.volume_crank_slider(
        theta_array, ctx["head"], ctx["porosity"], ctx["tdc_ref"])
    dV_c, dV_e = Kin.volume_derivatives_crank_slider(theta_array, ctx["tdc_ref"])
    return V_c, V_e, dV_c, dV_e


# ──────────────────────────────────────────────────────────────────────
#  4. ODE RIGHT-HAND SIDE
# ──────────────────────────────────────────────────────────────────────

def _adiabatic_rhs(theta: float,
                   y: NDArray[np.float64],
                   ctx: dict,
                   ) -> NDArray[np.float64]:
    """Compute the derivative dy/dtheta for the Urieli adiabatic ODE system.

    The two interface temperatures TCK and THE are updated at the end
    of the function according to the upwind direction of mass flow,
    then carried over to the next RK4 evaluation.
    """
    Tc, Te = y[_TC], y[_TE]
    TCK, THE = y[_TCK], y[_THE]

    # Volumes and their angular derivatives (from the selected volume model)
    theta_arr = np.array([theta])
    V_c_arr, V_e_arr, dV_c_arr, dV_e_arr = _volumes(theta_arr, ctx)

    V_c, V_e = float(V_c_arr[0]), float(V_e_arr[0])
    dV_c, dV_e = float(dV_c_arr[0]), float(dV_e_arr[0])

    # Constant volumes and temperatures
    V_k, V_r, V_h = ctx["V_k"], ctx["V_r"], ctx["V_h"]
    T_k, T_r, T_h = ctx["T_k"], ctx["T_r"], ctx["T_h"]
    gamma = ctx["gamma"]
    R     = ctx["R"]
    mR    = ctx["mR"]

    # Pressure from mass conservation (ideal gas, total inventory fixed)
    S = V_c / Tc + V_k / T_k + V_r / T_r + V_h / T_h + V_e / Te
    P = mR / S

    # Pressure derivative (Urieli eq. 3.21, using carried-over TCK and THE)
    top    = -P * (dV_c / TCK + dV_e / THE)
    bottom = (V_c / TCK
              + gamma * (V_k / T_k + V_r / T_r + V_h / T_h)
              + V_e / THE)
    dP = gamma * top / bottom

    # Mass derivatives in adiabatic cells
    dMC = (P * dV_c + V_c * dP / gamma) / (R * TCK)
    dME = (P * dV_e + V_e * dP / gamma) / (R * THE)

    # Mass flows at the four cell interfaces
    GACK = -dMC              # from c to k (positive if mass leaves c)
    GAHE =  dME              # from h to e (positive if mass enters e)
    dpop = dP / P
    MK = P * V_k / (R * T_k);  dMK = MK * dpop
    MR = P * V_r / (R * T_r);  dMR = MR * dpop
    MH = P * V_h / (R * T_h)
    GAKR = GACK - dMK
    GARH = GAHE + (MH * dpop - dMR)   # from r to h

    # Working-space temperature derivatives (Urieli eq. 3.26)
    MC = P * V_c / (R * Tc)
    ME = P * V_e / (R * Te)
    dTC = Tc * (dpop + dV_c / V_c - dMC / MC)
    dTE = Te * (dpop + dV_e / V_e - dME / ME)

    # Energy derivatives (HX heat fluxes and work in each space)
    cv = R / (gamma - 1.0)
    cp = gamma * cv
    dQK = V_k * dP * cv / R - cp * (TCK * GACK - T_k * GAKR)
    dQR = V_r * dP * cv / R - cp * (T_k * GAKR - T_h * GARH)
    dQH = V_h * dP * cv / R - cp * (T_h * GARH - THE * GAHE)
    dWC = P * dV_c
    dWE = P * dV_e

    # Update upwind interface temperatures based on current flow direction
    # (these are NOT integrated; they replace the previous carry-over values).
    TCK_new = Tc if GACK > 0 else T_k
    THE_new = T_h if GAHE > 0 else Te

    dy = np.zeros(_STATE_SIZE)
    dy[_TC] = dTC;  dy[_TE] = dTE
    dy[_QK] = dQK;  dy[_QR] = dQR;  dy[_QH] = dQH
    dy[_WC] = dWC;  dy[_WE] = dWE
    dy[_TCK] = TCK_new       # carry-over: assigned, not integrated
    dy[_THE] = THE_new
    return dy


# ──────────────────────────────────────────────────────────────────────
#  5. CUSTOM RK4 WITH CARRY-OVER
# ──────────────────────────────────────────────────────────────────────

def _rk4_step(theta: float,
              y: NDArray[np.float64],
              dtheta: float,
              ctx: dict,
              ) -> tuple[float, NDArray[np.float64]]:
    """One classical RK4 step with TCK/THE propagated as direct carry-over.

    The 7 integrated variables follow the standard RK4 update, while
    indices 7 and 8 (TCK, THE) are assigned the most recent values
    returned by the RHS evaluations, not averaged.
    """
    y0 = y.copy()

    k1 = _adiabatic_rhs(theta, y, ctx)
    y[:7] = y0[:7] + 0.5 * dtheta * k1[:7]
    y[7], y[8] = k1[7], k1[8]
    k2 = _adiabatic_rhs(theta + 0.5 * dtheta, y, ctx)
    y[:7] = y0[:7] + 0.5 * dtheta * k2[:7]
    y[7], y[8] = k2[7], k2[8]
    k3 = _adiabatic_rhs(theta + 0.5 * dtheta, y, ctx)
    y[:7] = y0[:7] + dtheta * k3[:7]
    y[7], y[8] = k3[7], k3[8]
    k4 = _adiabatic_rhs(theta + dtheta, y, ctx)

    # Final integrated state: weighted average of the four slopes
    dy_avg = (k1 + 2.0 * (k2 + k3) + k4) / 6.0
    y[:7] = y0[:7] + dtheta * dy_avg[:7]
    y[7], y[8] = k4[7], k4[8]   # last carry-over wins

    return theta + dtheta, y


# ──────────────────────────────────────────────────────────────────────
#  6. MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

def compute_state(theta_rad_out: NDArray[np.float64],
                  T_compression_K: float,
                  T_expansion_K: float,
                  p_mean_Pa: float,
                  head: int = 1,
                  porosity_percent: int = 62,
                  tdc_ref: TDCReference = Cfg.TDC_REFERENCE_IDEAL,
                  volume_model: str = "crank_slider",
                  tol_K: float = 1.0,
                  max_iter: int = 20,
                  ) -> AdiabaticResult:
    """Compute the full thermodynamic state under Urieli's adiabatic model.

    The solver iterates whole-cycle integrations until the boundary
    temperatures converge: |Tc(0) - Tc(2*pi)| + |Te(0) - Te(2*pi)| < tol_K.

    Parameters
    ----------
    theta_rad_out : array-like
        Output crank-angle grid (rad). Need not be uniform.
    T_compression_K, T_expansion_K : float
        Structural wall temperatures (sink and source in cooling mode).
    p_mean_Pa : float
        Target cycle-mean pressure used to calibrate helium inventory.
    head, porosity_percent : int
        Engine configuration.
    tdc_ref : TDCReference
        Angular convention (used only by the crank-slider volume model).
    volume_model : {"crank_slider", "fourier"}
        Kinematic basis for the working-space volumes. "fourier" keeps the
        model on the same Sage representation as the experimental pipeline.
    tol_K : float
        Cyclic convergence tolerance.
    max_iter : int
        Maximum number of cycles to integrate.

    Returns
    -------
    AdiabaticResult
        Dict with P, T_c, T_e at the requested output angles, plus diagnostics.
    """
    # Build the context dictionary shared by all RHS evaluations
    V_k = Cfg.V_COOLER_HX_M3
    V_h = Cfg.get_heater_volume(head)
    V_r = Cfg.get_regenerator_volume(porosity_percent)

    T_k = T_compression_K          # cooler-side wall (warm in cooling mode)
    T_h = T_expansion_K            # heater-side wall (cold in cooling mode)
    T_r = (T_h - T_k) / np.log(T_h / T_k)

    ctx = {
        "head": head,
        "porosity": porosity_percent,
        "tdc_ref": tdc_ref,
        "volume_model": volume_model,
        "V_k": V_k, "V_r": V_r, "V_h": V_h,
        "T_k": T_k, "T_r": T_r, "T_h": T_h,
        "gamma": Cfg.GAMMA_HELIUM,
        "R": Cfg.R_HELIUM_J_KG_K,
        "mR": 0.0,     # filled below after first calibration pass
    }

    # Calibrate mR so that the cycle-mean pressure matches p_mean_Pa.
    # Use one Schmidt-style pass on a coarse grid as initial estimate.
    theta_calib = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    V_c_calib_arr, V_e_calib_arr, _, _ = _volumes(theta_calib, ctx)
    S_calib = (V_c_calib_arr / T_k + V_e_calib_arr / T_h
               + V_k / T_k + V_r / T_r + V_h / T_h)
    inv_S_mean = np.mean(1.0 / S_calib)
    ctx["mR"] = p_mean_Pa / inv_S_mean

    # Initial state: walls at their structural temperatures
    y = np.zeros(_STATE_SIZE)
    y[_TC] = T_k
    y[_TE] = T_h
    y[_TCK] = T_k        # initial upwind = wall T (no flow info yet)
    y[_THE] = T_h

    # Cyclic-convergence loop
    n_grid = 360
    dtheta = 2.0 * np.pi / n_grid
    closure_error = np.inf
    n_iter = 0
    converged = False

    for n_iter in range(1, max_iter + 1):
        Tc_start, Te_start = y[_TC], y[_TE]
        # Reset cycle accumulators (Q and W) before each new integration
        y[_QK] = y[_QR] = y[_QH] = y[_WC] = y[_WE] = 0.0

        theta = 0.0
        for _ in range(n_grid):
            theta, y = _rk4_step(theta, y, dtheta, ctx)

        closure_error = abs(y[_TC] - Tc_start) + abs(y[_TE] - Te_start)
        if closure_error < tol_K:
            converged = True
            break

    # ── Final integration pass: record P, T_c, T_e at every step ──
    # Reset and integrate one more full cycle with output recording.
    y[_QK] = y[_QR] = y[_QH] = y[_WC] = y[_WE] = 0.0

    dense_theta = np.zeros(n_grid + 1)
    dense_P     = np.zeros(n_grid + 1)
    dense_Tc    = np.zeros(n_grid + 1)
    dense_Te    = np.zeros(n_grid + 1)

    # Record initial point
    theta = 0.0
    V_c_arr, V_e_arr, _, _ = _volumes(np.array([theta]), ctx)
    V_c, V_e = float(V_c_arr[0]), float(V_e_arr[0])
    S0 = V_c / y[_TC] + V_k / T_k + V_r / T_r + V_h / T_h + V_e / y[_TE]
    dense_P[0] = ctx["mR"] / S0
    dense_Tc[0] = y[_TC]
    dense_Te[0] = y[_TE]

    for i in range(1, n_grid + 1):
        theta, y = _rk4_step(theta, y, dtheta, ctx)
        V_c_arr, V_e_arr, _, _ = _volumes(np.array([theta]), ctx)
        V_c, V_e = float(V_c_arr[0]), float(V_e_arr[0])
        S = V_c / y[_TC] + V_k / T_k + V_r / T_r + V_h / T_h + V_e / y[_TE]
        dense_theta[i] = theta
        dense_P[i]  = ctx["mR"] / S
        dense_Tc[i] = y[_TC]
        dense_Te[i] = y[_TE]

    # Interpolate outputs at the requested (possibly unsorted) angles.
    theta_out_mod = np.mod(theta_rad_out, 2.0 * np.pi)
    P_out  = np.interp(theta_out_mod, dense_theta, dense_P,  period=2.0 * np.pi)
    Tc_out = np.interp(theta_out_mod, dense_theta, dense_Tc, period=2.0 * np.pi)
    Te_out = np.interp(theta_out_mod, dense_theta, dense_Te, period=2.0 * np.pi)

    return AdiabaticResult(
        P              = P_out,
        T_c            = Tc_out,
        T_e            = Te_out,
        n_iterations   = n_iter,
        closure_error_K= closure_error,
        converged      = converged,
    )
