# SlowDataProcessor.py
# Extraction of steady-state boundary conditions from the slow-data log.
#
# The slow-data channels (cold-head temperature, cooling-water temperatures,
# rotational speed, electric power and water flow) are recorded at low rate
# (1 Hz) during the whole run, from cold start-up to steady operation. The
# thermodynamic models, however, assume constant boundary temperatures. This
# module identifies the steady-state window -- where the cold head has stopped
# cooling and the machine is running at speed -- and averages the channels over
# that window, reproducing the procedure used to report the reference values.
#
# Following the acquisition philosophy of the ecosystem, LabVIEW writes the raw
# time series and the averaging is performed here, in Python.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Mapping from distinctive header substrings (Norwegian/English) to canonical
# channel names. Matching is case-insensitive and order-sensitive: more
# specific keys are tested first.
_COLUMN_KEYWORDS = [
    ("topp",        "T_cold_head_C"),
    ("cold_head",   "T_cold_head_C"),
    ("boks",        "T_thermobox_C"),
    ("inn",         "T_water_in_C"),
    ("water_in",    "T_water_in_C"),
    ("ut",          "T_water_out_C"),
    ("water_out",   "T_water_out_C"),
    ("turtall",     "rpm"),
    ("rpm",         "rpm"),
    ("dreiemoment", "torque_pct"),
    ("torque",      "torque_pct"),
    ("effekt",      "power_electric_W"),
    ("power",       "power_electric_W"),
    ("l/min",       "water_flow_Lmin"),
    ("flow",        "water_flow_Lmin"),
]


@dataclass
class BoundaryConditions:
    """Steady-state boundary conditions averaged from the slow-data log."""
    T_sink_K: float            # cooling-water inlet (hot focus)
    T_source_K: float          # cold-head temperature (cold focus)
    rpm: float                 # mean rotational speed
    power_electric_W: float    # mean electric power
    water_flow_Lmin: float     # mean cooling-water flow
    water_delta_T_K: float     # outlet minus inlet water temperature
    n_steady_samples: int      # number of samples in the steady window


# ──────────────────────────────────────────────────────────────────────
#  1. RAW LOG READING
# ──────────────────────────────────────────────────────────────────────

def _read_lines_robust(path: Path) -> list[str]:
    """Read all lines, tolerating the Norwegian (Latin-1) encoding."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as fh:
                return fh.readlines()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.readlines()


def _map_columns(header: list[str]) -> dict[int, str]:
    """Map each raw header column index to a canonical channel name."""
    mapping: dict[int, str] = {}
    for idx, raw in enumerate(header):
        low = raw.strip().lower()
        for keyword, canonical in _COLUMN_KEYWORDS:
            if keyword in low:
                mapping[idx] = canonical
                break
    return mapping


def read_slow_log(path: str | Path) -> pd.DataFrame:
    """Read a raw slow-data log into a DataFrame with canonical columns.

    The leading time-stamp column is ignored, the Norwegian decimal comma is
    honoured, and channels are renamed by keyword so the reader is robust to
    minor header variations.
    """
    path = Path(path)
    lines = [ln.strip() for ln in _read_lines_robust(path) if ln.strip()]
    header = lines[0].split(";")
    mapping = _map_columns(header)

    records: list[dict[str, float]] = []
    for line in lines[1:]:
        fields = line.split(";")
        if len(fields) < len(header):
            continue
        row: dict[str, float] = {}
        for idx, canonical in mapping.items():
            token = fields[idx].strip().replace(",", ".")
            try:
                row[canonical] = float(token)
            except ValueError:
                row[canonical] = float("nan")
        records.append(row)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────
#  2. STEADY-STATE DETECTION
# ──────────────────────────────────────────────────────────────────────

def detect_steady_state(df: pd.DataFrame,
                       rpm_threshold: float = 1000.0,
                       slope_threshold_C: float = 0.5,
                       ) -> np.ndarray:
    """Return a boolean mask of the samples in steady-state operation.

    A sample is in steady state when the machine is running above a speed
    threshold and the cold-head temperature has essentially stopped changing
    (absolute slope below a threshold). The cooling transient and the idle
    samples before start-up are therefore excluded.
    """
    rpm = np.abs(df["rpm"].to_numpy())
    T_cold = df["T_cold_head_C"].to_numpy()
    slope = np.abs(np.gradient(T_cold))

    running = rpm > rpm_threshold
    stable = slope < slope_threshold_C
    return running & stable


# ──────────────────────────────────────────────────────────────────────
#  3. BOUNDARY-CONDITION EXTRACTION
# ──────────────────────────────────────────────────────────────────────

def extract_boundary_conditions(df: pd.DataFrame,
                               rpm_threshold: float = 1000.0,
                               slope_threshold_C: float = 0.5,
                               ) -> BoundaryConditions:
    """Average the slow-data channels over the steady-state window.

    The cooling-water inlet temperature defines the heat-rejection (sink)
    temperature; the cold-head temperature defines the cold-source temperature.
    Both are converted to kelvin.
    """
    mask = detect_steady_state(df, rpm_threshold, slope_threshold_C)
    if mask.sum() == 0:
        raise ValueError("No steady-state samples detected in slow-data log")

    steady = df[mask]
    T_source_C = float(steady["T_cold_head_C"].mean())
    T_sink_C = float(steady["T_water_in_C"].mean())
    T_water_out_C = float(steady["T_water_out_C"].mean())

    return BoundaryConditions(
        T_sink_K=T_sink_C + 273.15,
        T_source_K=T_source_C + 273.15,
        rpm=abs(float(steady["rpm"].mean())),
        power_electric_W=abs(float(steady["power_electric_W"].mean())),
        water_flow_Lmin=abs(float(steady["water_flow_Lmin"].mean())),
        water_delta_T_K=T_water_out_C - T_sink_C,
        n_steady_samples=int(mask.sum()),
    )


def process_slow_log(path: str | Path,
                    rpm_threshold: float = 1000.0,
                    slope_threshold_C: float = 0.5,
                    ) -> BoundaryConditions:
    """Read a raw slow-data log and return its steady-state boundary conditions."""
    df = read_slow_log(path)
    return extract_boundary_conditions(df, rpm_threshold, slope_threshold_C)


def boundary_from_dict(slow: dict) -> BoundaryConditions:
    """Build boundary conditions from pre-averaged single-value slow data.

    Some unified files carry the slow channels already reduced to one value
    each in their metadata-style block, rather than as a raw time series. This
    helper maps those values to the boundary conditions, recognising both the
    canonical English keys and the Norwegian originals.
    """
    def _find(*needles: str) -> float:
        for key, value in slow.items():
            low = key.lower()
            if all(n in low for n in needles):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return float("nan")

    T_source_C = _find("cold", "head")
    if np.isnan(T_source_C):
        T_source_C = _find("topp")
    T_sink_C = _find("water", "in")
    if np.isnan(T_sink_C):
        T_sink_C = _find("inn")
    T_out_C = _find("water", "out")
    if np.isnan(T_out_C):
        T_out_C = _find("ut")

    return BoundaryConditions(
        T_sink_K=T_sink_C + 273.15,
        T_source_K=T_source_C + 273.15,
        rpm=abs(_find("rpm")) if not np.isnan(_find("rpm")) else abs(_find("turtall")),
        power_electric_W=abs(_find("power")) if not np.isnan(_find("power")) else abs(_find("effekt")),
        water_flow_Lmin=abs(_find("flow")) if not np.isnan(_find("flow")) else abs(_find("l/min")),
        water_delta_T_K=T_out_C - T_sink_C,
        n_steady_samples=1,
    )
