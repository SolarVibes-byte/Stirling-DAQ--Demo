# MathematicalProcessor.py
# Orchestrator of the experimental-validation pipeline.
#
# This module coordinates the whole processing chain for one operating point.
# It performs no physical calculation itself: every quantity is delegated to
# the specialised module that owns it, in keeping with the single-responsibility
# design. Its role is to route the data through the correct path, pair the
# experimental pressure with the reference volume, and evaluate the comparison
# models on the same grid.
#
# Two data layouts are handled, distinguished automatically from the parsed
# fast-data format rather than from external labels:
#   - Historical files already reduced by LabVIEW (angle in degrees): the
#     pressure is used at its recorded angles, with no angular reprocessing,
#     reproducing the supervisor's procedure.
#   - Unified v2.0 files with raw encoder pulses: the full preprocessor is run
#     to reconstruct, align and ensemble-average the cycle.
# The absolute gas pressure always follows the fixed sensor convention
# (Kistler gauge + mean Danfoss static + atmospheric), justified by the
# piezoelectric nature of the dynamic sensor. The volume is taken from the
# Sage truncated-Fourier model, the reference of the work; the crank-slider
# volume is also returned, but only for model-to-model comparison.

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import Configuration as Cfg
import Parsers as Prs
from Parsers import FastFormat
import Preprocessor as Pre
from Preprocessor import SyncMethod
import SlowDataProcessor as SDP
from SlowDataProcessor import BoundaryConditions
import Metrics as Met
import Kinematics as Kin
import ThermodynamicsSchmidt as Sch
import ThermodynamicsAdiabatic as Ada
from Configuration import TDCReference

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Complete result of processing one operating point."""
    theta_deg: np.ndarray              # angular grid of the averaged cycle
    V_fourier_m3: np.ndarray           # Sage Fourier total volume (reference)
    V_crank_slider_m3: np.ndarray      # crank-slider total volume (comparison)
    P_experimental_bar: np.ndarray     # reconstructed experimental pressure
    P_schmidt_bar: np.ndarray          # Schmidt model pressure
    P_adiabatic_bar: np.ndarray        # adiabatic model pressure
    W_experimental_J: float            # |indicated work| of the measured loop
    W_schmidt_J: float
    W_adiabatic_J: float
    operating_mode: str                # inferred from the loop sign
    T_sink_K: float
    T_source_K: float
    boundary: BoundaryConditions | None
    case_name: str
    data_path: str                     # 'historical' or 'v2_pulses'
    metrics: "Met.PerformanceMetrics | None" = None   # refrigeration-mode KPIs


# ──────────────────────────────────────────────────────────────────────
#  EXPERIMENTAL CYCLE (format-dependent path)
# ──────────────────────────────────────────────────────────────────────

def _experimental_cycle(parsed: Prs.ParsedFile,
                       sync_method: SyncMethod,
                       n_discard: int,
                       ) -> tuple[np.ndarray, np.ndarray, str]:
    """Return one averaged experimental cycle as (theta_deg, P_abs_bar, path).

    The path is chosen from the fast-data format: pulse files are fully
    preprocessed, whereas degree files (already reduced by LabVIEW) are used at
    their recorded angles after a plain cycle average, with no angular shift.
    """
    if parsed.fast_format == FastFormat.PULSES_RAW:
        clean = Pre.preprocess(parsed, sync_method=sync_method, n_discard=n_discard)
        return clean["theta_deg"].to_numpy(), clean["pressure_bar"].to_numpy(), "v2_pulses"

    # Historical degrees: average whole cycles on the recorded angles
    fast = parsed.fast_data
    theta = np.mod(fast["theta_deg"].to_numpy(), 360.0)
    P_abs = (fast["P_kistler_gauge_bar"].to_numpy()
             + float(fast["P_danfoss_static_bar"].mean())
             + Cfg.ATMOSPHERIC_PRESSURE_BAR)

    grid = np.arange(Cfg.ANGULAR_POINTS_PER_CYCLE) * (360.0 / Cfg.ANGULAR_POINTS_PER_CYCLE)
    order = np.argsort(theta)
    P_grid = np.interp(grid, theta[order], P_abs[order], period=360.0)
    return grid, P_grid, "historical"


# ──────────────────────────────────────────────────────────────────────
#  BOUNDARY-CONDITION RESOLUTION  (hybrid: infer, else label, else fail)
# ──────────────────────────────────────────────────────────────────────

def _resolve_boundary(parsed: Prs.ParsedFile,
                     slow_path: str | Path | None,
                     T_sink_K: float | None,
                     T_source_K: float | None,
                     ) -> tuple[float, float, BoundaryConditions | None]:
    if slow_path is not None:
        bc = SDP.process_slow_log(slow_path)
        logger.info("Boundary temperatures from slow-data log: "
                    "T_sink=%.1f K, T_source=%.1f K (%d steady samples)",
                    bc.T_sink_K, bc.T_source_K, bc.n_steady_samples)
        return bc.T_sink_K, bc.T_source_K, bc

    if parsed.slow_data:
        bc = SDP.boundary_from_dict(parsed.slow_data)
        if not (np.isnan(bc.T_sink_K) or np.isnan(bc.T_source_K)):
            logger.info("Boundary temperatures from embedded slow block: "
                        "T_sink=%.1f K, T_source=%.1f K", bc.T_sink_K, bc.T_source_K)
            return bc.T_sink_K, bc.T_source_K, bc

    if T_sink_K is not None and T_source_K is not None:
        logger.warning("No slow data available; using fixed boundary "
                       "temperatures T_sink=%.1f K, T_source=%.1f K",
                       T_sink_K, T_source_K)
        return T_sink_K, T_source_K, None

    raise ValueError("Cannot resolve boundary temperatures: provide a "
                     "slow-data log, an embedded slow block, or explicit "
                     "T_sink_K and T_source_K.")


# ──────────────────────────────────────────────────────────────────────
#  MODE INFERENCE  (from the loop sign, verified against any label)
# ──────────────────────────────────────────────────────────────────────

def _signed_work(P_bar: np.ndarray, V_m3: np.ndarray) -> float:
    """Closed-loop work (signed): negative is counter-clockwise (cooling)."""
    P = np.append(P_bar, P_bar[0]) * 1e5
    V = np.append(V_m3, V_m3[0])
    return float(np.trapezoid(P, V))


def _infer_mode(signed_work_J: float, parsed: Prs.ParsedFile) -> str:
    """Infer the operating mode from the loop sign and cross-check the label."""
    mode = "cooling" if signed_work_J < 0 else "engine"
    label = str(parsed.metadata.get("Mode", "")).lower()
    if label and label != mode:
        logger.warning("Declared mode '%s' disagrees with the loop sign "
                       "(inferred '%s'); trusting the data.", label, mode)
    return mode


# ──────────────────────────────────────────────────────────────────────
#  ORCHESTRATION
# ──────────────────────────────────────────────────────────────────────

def process_case(fast_path: str | Path,
                slow_path: str | Path | None = None,
                sync_method: SyncMethod = SyncMethod.BDC_180,
                head: int = 1,
                porosity_percent: int = 62,
                T_sink_K: float | None = None,
                T_source_K: float | None = None,
                n_discard: int = Cfg.N_CYCLES_DISCARDED_INIT,
                ) -> ProcessingResult:
    """Process one operating point into an experiment-vs-model result.

    The fast-data format selects the processing path automatically; the
    boundary temperatures are resolved by the hybrid rule; the reference volume
    is the Sage Fourier model and the crank-slider volume is provided for
    comparison only.
    """
    fast_path = Path(fast_path)
    parsed = Prs.parse_file(fast_path)

    # 1. Experimental cycle (path chosen from the data format)
    theta_deg, P_exp_bar, data_path = _experimental_cycle(
        parsed, sync_method, n_discard)
    theta_rad = np.deg2rad(theta_deg)

    # 2. Boundary temperatures (hybrid resolution)
    T_sink, T_source, boundary = _resolve_boundary(
        parsed, slow_path, T_sink_K, T_source_K)

    # 3. Reference volume (Sage Fourier). The compression and expansion parts
    #    feed the comparison models; the crank-slider total is kept only as a
    #    secondary comparison curve, never mixed into the model pressures.
    V_c, V_e, V_fourier = Kin.volume_fourier(
        theta_rad, head=head, porosity_percent=porosity_percent)
    _, _, V_crank = Kin.volume_crank_slider(
        theta_rad, head=head, porosity_percent=porosity_percent,
        tdc_ref=TDCReference.DISPLACER)

    # 4. Comparison models, calibrated to the experimental mean pressure and
    #    built entirely on the Fourier volume decomposition.
    p_mean_Pa = float(np.mean(P_exp_bar)) * 1e5
    P_sch = Sch.compute_pressure(V_c, V_e, T_sink, T_source, p_mean_Pa,
                                 head=head, porosity_percent=porosity_percent) / 1e5
    ada = Ada.compute_state(theta_rad, T_sink, T_source, p_mean_Pa,
                            head=head, porosity_percent=porosity_percent,
                            volume_model="fourier")
    P_ada = ada["P"] / 1e5

    # 5. Indicated work of each loop, all integrated against the Fourier volume.
    W_exp_signed = _signed_work(P_exp_bar, V_fourier)
    mode = _infer_mode(W_exp_signed, parsed)

    # 6. Refrigeration-mode performance metrics (global balance), when the slow
    #    data provides the coolant flow and the electric power needed to close it.
    metrics = None
    if boundary is not None and not (np.isnan(boundary.water_flow_Lmin)
                                     or np.isnan(boundary.power_electric_W)):
        metrics = Met.compute_performance(
            W_ind_J=abs(W_exp_signed),
            rpm=boundary.rpm,
            T_sink_K=T_sink,
            T_source_K=T_source,
            water_flow_Lmin=boundary.water_flow_Lmin,
            water_delta_T_K=boundary.water_delta_T_K,
            power_electric_W=boundary.power_electric_W,
        )

    return ProcessingResult(
        theta_deg=theta_deg,
        V_fourier_m3=V_fourier,
        V_crank_slider_m3=V_crank,
        P_experimental_bar=P_exp_bar,
        P_schmidt_bar=P_sch,
        P_adiabatic_bar=P_ada,
        W_experimental_J=abs(W_exp_signed),
        W_schmidt_J=abs(_signed_work(P_sch, V_fourier)),
        W_adiabatic_J=abs(_signed_work(P_ada, V_fourier)),
        operating_mode=mode,
        T_sink_K=T_sink,
        T_source_K=T_source,
        boundary=boundary,
        case_name=fast_path.stem,
        data_path=data_path,
        metrics=metrics,
    )
