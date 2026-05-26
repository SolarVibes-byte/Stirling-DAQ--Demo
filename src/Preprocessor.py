# Preprocessor.py
# Angular alignment and ensemble averaging of the raw fast-data signal.
#
# The module turns the raw pressure samples produced by Parsers into a
# single clean cycle on a synthetic angular grid, ready for the P-V
# reconstruction. It is built as an explicit pipeline of independent steps
# so that each acquisition pathology is addressed (and tested) in isolation:
#   reconstruct_angle -> apply_encoder_offset -> synchronize
#   -> ensemble_average -> compute_absolute_pressure
#
# Two synchronisation references are provided:
#   BDC_180     (default) cuts the cycles at the displacer bottom dead centre
#               (true 180 deg), discarding the initial half-revolution that
#               contains the random recording start and the start-up transient.
#   TDC_MARKER  cuts the cycles at the displacer top dead centre (true 0 deg),
#               taken from the offset-corrected hardware marker.
# Both produce a cycle on the same true-angle grid; they are compared on
# synthetic data to assess which recovers the reference cycle more faithfully.

from __future__ import annotations
from enum import Enum
import logging
import numpy as np
import pandas as pd

import Configuration as Cfg
from Parsers import ParsedFile, FastFormat

logger = logging.getLogger(__name__)


class SyncMethod(Enum):
    """Cycle-synchronisation reference."""
    BDC_180    = "bdc_180"      # cut at displacer BDC (true 180 deg)
    TDC_MARKER = "tdc_marker"   # cut at displacer TDC (true 0 deg)


# ──────────────────────────────────────────────────────────────────────
#  1. ANGLE RECONSTRUCTION
# ──────────────────────────────────────────────────────────────────────

def reconstruct_angle(fast: pd.DataFrame,
                     fast_format: FastFormat,
                     ) -> pd.DataFrame:
    """Add a continuous encoder angle measured from the recording reference.

    For the pulses-raw format the angle is rebuilt from the pulse counter,
    taking the first TDC marker as the origin; samples before it carry NaN
    so the random recording start can be cropped. For the degrees-direct
    format the angle is unwrapped from the first sample.
    """
    out = fast.copy()

    if fast_format == FastFormat.DEGREES_DIRECT:
        theta = out["theta_deg"].to_numpy(dtype=float)
        # Unwrap into a monotonic continuous angle from the first sample
        cont = np.unwrap(np.deg2rad(theta))
        cont = np.rad2deg(cont) - np.rad2deg(np.deg2rad(theta[0]))
        out["angle_enc"] = cont
        return out

    pulses = out["angle_pulses"].to_numpy(dtype=float)
    markers = out["tdc_marker"].to_numpy(dtype=int)

    angle_enc = np.full(len(pulses), np.nan)
    marker_pos = np.where(markers == 1)[0]
    if marker_pos.size > 0:
        p0 = pulses[marker_pos[0]]
        valid = pulses >= p0
        angle_enc[valid] = (pulses[valid] - p0) * Cfg.ENCODER_DEG_PER_PULSE

    out["angle_enc"] = angle_enc
    return out


# ──────────────────────────────────────────────────────────────────────
#  2. ENCODER OFFSET
# ──────────────────────────────────────────────────────────────────────

def apply_encoder_offset(fast: pd.DataFrame,
                        offset_deg: float = Cfg.ENCODER_OFFSET_DEG,
                        ) -> pd.DataFrame:
    """Convert the encoder angle into the true crank angle.

    The continuous true angle is the encoder angle minus the mounting offset;
    the wrapped true angle is stored as 'theta_deg' for the P-V evaluation.
    With a zero offset the operation is a plain copy.
    """
    out = fast.copy()
    angle_true = out["angle_enc"].to_numpy() - offset_deg
    out["angle_true"] = angle_true
    out["theta_deg"] = np.mod(angle_true, 360.0)
    return out


# ──────────────────────────────────────────────────────────────────────
#  3. SYNCHRONISATION (cycle windowing)
# ──────────────────────────────────────────────────────────────────────

def synchronize(fast: pd.DataFrame,
               method: SyncMethod = SyncMethod.BDC_180,
               n_discard: int = Cfg.N_CYCLES_DISCARDED_INIT,
               ) -> pd.DataFrame:
    """Assign a cycle index from the chosen reference and drop the rest.

    The continuous true angle is split into 360-degree windows starting at
    the reference (0 deg for the TDC marker, 180 deg for the BDC). Samples
    before the reference (including the random-start remainder and, for the
    BDC method, the initial half-revolution) are discarded, together with the
    first ``n_discard`` whole cycles and any incomplete cycle at the end.
    """
    out = fast.dropna(subset=["angle_true"]).copy()
    angle = out["angle_true"].to_numpy()

    ref = 180.0 if method == SyncMethod.BDC_180 else 0.0
    rel = angle - ref
    cycle_idx = np.floor(rel / 360.0).astype(int)
    cycle_idx[rel < 0] = -1
    out["cycle_idx"] = cycle_idx

    out = out[out["cycle_idx"] >= n_discard].copy()

    if len(out) == 0:
        logger.warning("No complete cycles left after discarding %d initial "
                       "cycles; the recording may be too short for this "
                       "n_discard value.", n_discard)
        return out.reset_index(drop=True)

    # Keep only complete cycles (a partial final window would bias the average)
    counts = out.groupby("cycle_idx").size()
    if len(counts) > 0:
        full_len = int(counts.max())
        complete = counts[counts >= full_len].index
        out = out[out["cycle_idx"].isin(complete)].copy()

    return out.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
#  3b. MISSING-DATA HANDLING (NaN in the pressure signals)
# ──────────────────────────────────────────────────────────────────────

class TooManyCorruptCyclesError(ValueError):
    """Raised when too few clean cycles remain to trust the result."""


def handle_missing_data(fast: pd.DataFrame,
                       max_nan_fraction_per_cycle: float = Cfg.MAX_NAN_FRACTION_PER_CYCLE,
                       min_clean_cycles: int = Cfg.MIN_CLEAN_CYCLES,
                       ) -> pd.DataFrame:
    """Clean missing pressure samples following a severity-tiered strategy.

    The acquisition can drop samples (sensor glitch, comms loss), which appear
    as NaN in the pressure columns. They are handled by gravity, per cycle:
      - a few isolated NaN  -> linearly interpolated within the cycle, since
        the neighbouring samples are valid;
      - a heavily corrupted cycle (NaN fraction above the threshold) -> the
        whole cycle is discarded, as interpolation would invent too much;
      - too few clean cycles left overall -> the file is rejected by raising,
        so the orchestrator routes it to the failed directory.

    A constant offset between models is contable, not physical; here, by
    contrast, the NaN are genuine gaps and must be repaired or dropped.
    """
    cols = ["P_kistler_gauge_bar", "P_danfoss_static_bar"]
    kept: list[pd.DataFrame] = []
    dropped = 0

    for idx, g in fast.groupby("cycle_idx"):
        g = g.sort_values("theta_deg").copy()
        n = len(g)
        worst_fraction = max(
            float(g[c].isna().sum()) / n if n else 1.0 for c in cols)

        if worst_fraction > max_nan_fraction_per_cycle:
            dropped += 1
            logger.warning("Cycle %s dropped: %.0f%% missing pressure samples",
                           idx, worst_fraction * 100)
            continue

        # Interpolate isolated gaps within the cycle (periodic, both ends)
        for c in cols:
            s = g[c]
            if s.isna().any():
                g[c] = (s.interpolate(method="linear", limit_direction="both")
                         .to_numpy())
        kept.append(g)

    if dropped:
        logger.info("Missing-data handling: %d cycle(s) discarded", dropped)

    if len(kept) < min_clean_cycles:
        raise TooManyCorruptCyclesError(
            f"only {len(kept)} clean cycle(s) left after missing-data "
            f"handling, need at least {min_clean_cycles}")

    return pd.concat(kept, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
#  4. ENSEMBLE AVERAGING
# ──────────────────────────────────────────────────────────────────────

def ensemble_average(fast: pd.DataFrame,
                    n_points_out: int = Cfg.ANGULAR_POINTS_PER_CYCLE,
                    ) -> pd.DataFrame:
    """Resample every cycle onto a common grid and average across cycles.

    Each cycle is interpolated periodically onto the synthetic grid
    [0, 0.5, ..., 359.5] deg. This linear interpolation recovers the exact
    angles lost to the 3-degree discrete sampling and to the fractional
    encoder offset (including the exact reference point itself). Averaging
    the resampled cycles suppresses the random sensor noise by roughly the
    square root of the number of cycles retained.
    """
    grid = np.arange(n_points_out) * (360.0 / n_points_out)
    groups = [g for _, g in fast.groupby("cycle_idx")]

    kis_stack: list[np.ndarray] = []
    dan_stack: list[np.ndarray] = []
    for g in groups:
        theta = np.mod(g["theta_deg"].to_numpy(), 360.0)
        order = np.argsort(theta)
        theta_s = theta[order]
        kis_s = g["P_kistler_gauge_bar"].to_numpy()[order]
        dan_s = g["P_danfoss_static_bar"].to_numpy()[order]
        kis_stack.append(np.interp(grid, theta_s, kis_s, period=360.0))
        dan_stack.append(np.interp(grid, theta_s, dan_s, period=360.0))

    return pd.DataFrame({
        "theta_deg":            grid,
        "P_kistler_gauge_bar":  np.mean(kis_stack, axis=0),
        "P_danfoss_static_bar": np.mean(dan_stack, axis=0),
    })


# ──────────────────────────────────────────────────────────────────────
#  5. ABSOLUTE PRESSURE
# ──────────────────────────────────────────────────────────────────────

def compute_absolute_pressure(fast: pd.DataFrame,
                            p_atm_bar: float = Cfg.ATMOSPHERIC_PRESSURE_BAR,
                            ) -> pd.DataFrame:
    """Reconstruct the absolute gas pressure on the averaged cycle.

        P_gas_abs(theta) = P_kistler(theta) + mean(P_danfoss) + P_atm

    The static reference is averaged over a single whole cycle, so the
    non-integer-cycle bias seen at the parser stage no longer applies.
    """
    out = fast.copy()
    P_static_ref = float(np.mean(out["P_danfoss_static_bar"].to_numpy()))
    out["pressure_bar"] = (out["P_kistler_gauge_bar"].to_numpy()
                           + P_static_ref + p_atm_bar)
    return out


# ──────────────────────────────────────────────────────────────────────
#  6. ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────

def preprocess(parsed: ParsedFile,
             sync_method: SyncMethod = SyncMethod.BDC_180,
             offset_deg: float | None = None,
             n_discard: int = Cfg.N_CYCLES_DISCARDED_INIT,
             p_atm_bar: float = Cfg.ATMOSPHERIC_PRESSURE_BAR,
             ) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a parsed file.

    Returns a clean DataFrame with the columns 'theta_deg' (synthetic grid)
    and 'pressure_bar' (absolute, ensemble-averaged over the retained cycles).
    """
    if offset_deg is None:
        offset_deg = float(parsed.metadata.get("Encoder_Offset_deg",
                                               Cfg.ENCODER_OFFSET_DEG))

    fast = reconstruct_angle(parsed.fast_data, parsed.fast_format)
    fast = apply_encoder_offset(fast, offset_deg)
    fast = synchronize(fast, sync_method, n_discard)
    fast = handle_missing_data(fast)
    fast = ensemble_average(fast)
    fast = compute_absolute_pressure(fast, p_atm_bar)

    return fast[["theta_deg", "pressure_bar"]]
