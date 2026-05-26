# Parsers.py
# Raw acquisition-file reader for the Sigma 1-125A unified log format.
#
# The unified file groups three labelled blocks:
#   [METADATA]   — test configuration (mode, charge pressure, rpm, ...)
#   [SLOW_DATA]  — low-frequency channels for the energy balance
#   [FAST_DATA]  — high-frequency pressure signal for the P-V diagram
#
# Two FAST_DATA layouts are supported:
#   DEGREES_DIRECT — angle already in degrees (Norbert ISEC historical data)
#   PULSES_RAW     — raw encoder pulses + TDC marker (technician v2.0 format)
#
# The module only reads and interprets the file structure. The absolute gas
# pressure is NOT computed here: it requires averaging the static reference
# over an integer number of cycles, which only the Preprocessor can crop.
# The two pressure signals are therefore returned separately in raw form.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
#  1. DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────

class FastFormat(Enum):
    """Layout of the FAST_DATA block."""
    DEGREES_DIRECT = "degrees"   # Angle_deg; P_kistler; P_danfoss
    PULSES_RAW     = "pulses"    # Cycle; TDC; Pulses; Time; P_kistler; P_danfoss


@dataclass
class ParsedFile:
    """Structured content of a unified acquisition file."""
    metadata:    dict[str, str]   = field(default_factory=dict)
    slow_data:   dict[str, float] = field(default_factory=dict)
    fast_data:   pd.DataFrame     = field(default_factory=pd.DataFrame)
    fast_format: FastFormat       = FastFormat.DEGREES_DIRECT


# Slow-data keys whose sign only encodes the (reverse) rotation direction in
# cooling mode and must therefore be taken as magnitudes.
_SLOW_ABS_KEYS: tuple[str, ...] = (
    "rpm_measured", "torque_pct", "power_electric_w", "cooling_water_flow_lmin",
)


# ──────────────────────────────────────────────────────────────────────
#  2. LOW-LEVEL READING HELPERS
# ──────────────────────────────────────────────────────────────────────

def _read_lines_robust(path: Path) -> list[str]:
    """Read all lines, tolerating the Norwegian (Latin-1) encoding.

    Historical LabVIEW files use characters such as the Norwegian letter in
    "Kjolevann", stored in Latin-1 rather than UTF-8. The reader tries the
    UTF-8 variants first and falls back to Latin-1 so that no real file is
    rejected on a decoding error.
    """
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as fh:
                return fh.readlines()
        except UnicodeDecodeError:
            continue
    # Last resort: read replacing undecodable bytes
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.readlines()


def _detect_decimal(sample_lines: list[str]) -> str:
    """Detect the decimal separator (',' Norwegian or '.' international).

    A defensive guard against LabVIEW regional-format mistakes: the decimal
    mark that appears most often inside numeric fields is taken as active.
    """
    comma_decimals = 0
    dot_decimals = 0
    for line in sample_lines:
        for tok in line.split(";"):
            tok = tok.strip()
            if "," in tok and tok.replace(",", "").replace("-", "").isdigit():
                comma_decimals += 1
            if "." in tok and tok.replace(".", "").replace("-", "").isdigit():
                dot_decimals += 1
    return "," if comma_decimals >= dot_decimals else "."


def _to_float(token: str, decimal: str) -> float:
    """Convert a single numeric token to float, honouring the decimal mark.

    Non-numeric tokens such as clock timestamps (HH:MM:SS) found in the slow
    data log are returned as NaN instead of raising, so that a mixed column
    does not abort the whole parse.
    """
    token = token.strip()
    if decimal == ",":
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return float("nan")


def _split_blocks(lines: list[str]) -> dict[str, list[str]]:
    """Split the file lines into the labelled blocks they belong to.

    Lines before the first recognised header are ignored. A bare file with
    no headers (legacy Norbert data) is returned under a FAST_DATA key.
    """
    blocks: dict[str, list[str]] = {}
    current: str | None = None

    known = {"[METADATA]", "[SLOW_DATA]", "[FAST_DATA]"}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.upper() in known:
            current = line.upper().strip("[]")
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)

    if not blocks:
        blocks["FAST_DATA"] = [l.strip() for l in lines if l.strip()]
    return blocks


# ──────────────────────────────────────────────────────────────────────
#  3. BLOCK PARSERS
# ──────────────────────────────────────────────────────────────────────

def _parse_metadata(block: list[str]) -> dict[str, str]:
    """Parse the [METADATA] block into a key-value dictionary."""
    meta: dict[str, str] = {}
    for line in block:
        if ";" not in line:
            continue
        key, value = line.split(";", 1)
        meta[key.strip()] = value.strip()
    return meta


def _parse_slow_data(block: list[str], decimal: str) -> dict[str, float]:
    """Parse the [SLOW_DATA] block into key-value magnitudes.

    Supports both key;value pairs (one per line) and a two-row table
    (header + values). Physical magnitudes are taken in absolute value.
    """
    slow: dict[str, float] = {}

    if block and block[0].count(";") >= 3 and len(block) >= 2:
        headers = [h.strip() for h in block[0].split(";") if h.strip()]
        values  = [v.strip() for v in block[1].split(";") if v.strip()]
        pairs = list(zip(headers, values))
    else:
        pairs = [tuple(l.split(";", 1)) for l in block if ";" in l]

    for key, value in pairs:
        key = key.strip()
        try:
            val = _to_float(value, decimal)
        except ValueError:
            continue  # skip non-numeric fields such as timestamps
        if key.lower() in _SLOW_ABS_KEYS:
            val = abs(val)
        slow[key] = val
    return slow


def _detect_fast_format(header: list[str]) -> FastFormat:
    """Infer the FAST_DATA layout from its header columns."""
    hl = [h.lower() for h in header]
    if any("pulse" in h for h in hl) or any("tdc" in h for h in hl):
        return FastFormat.PULSES_RAW
    return FastFormat.DEGREES_DIRECT


def _identify_pressure_columns(df: pd.DataFrame,
                               candidates: list[str],
                               ) -> tuple[str, str]:
    """Identify the (kistler, danfoss) columns.

    Explicit names take priority: 'kistler' -> dynamic, 'danfoss' -> static.
    If the names are ambiguous (legacy Norwegian data), fall back to physics:
    the dynamic sensor has the larger standard deviation.
    """
    by_name_dyn = [c for c in candidates if "kistler" in c.lower()]
    by_name_sta = [c for c in candidates if "danfoss" in c.lower()]
    if by_name_dyn and by_name_sta:
        return by_name_dyn[0], by_name_sta[0]

    if len(candidates) != 2:
        raise ValueError(f"Expected two pressure columns, got {candidates}")
    c0, c1 = candidates
    if df[c0].std() >= df[c1].std():
        return c0, c1
    return c1, c0


def _parse_fast_data(block: list[str], decimal: str) -> tuple[pd.DataFrame, FastFormat]:
    """Parse the [FAST_DATA] block into a DataFrame with raw separated signals.

    Returned columns depend on the detected layout:
      DEGREES_DIRECT -> theta_deg, P_kistler_gauge_bar, P_danfoss_static_bar
      PULSES_RAW     -> cycle_count, tdc_marker, angle_pulses,
                        P_kistler_gauge_bar, P_danfoss_static_bar
    """
    header = [h.strip() for h in block[0].split(";") if h.strip()]
    fmt = _detect_fast_format(header)

    rows: list[list[float]] = []
    for line in block[1:]:
        fields = [f for f in line.split(";") if f.strip() != ""]
        if len(fields) < len(header):
            continue
        rows.append([_to_float(f, decimal) for f in fields[:len(header)]])
    table = pd.DataFrame(rows, columns=header)

    if fmt == FastFormat.DEGREES_DIRECT:
        angle_col = header[0]
        pressure_cols = header[1:3]
        kis_col, dan_col = _identify_pressure_columns(table, pressure_cols)
        out = pd.DataFrame({
            "theta_deg":            table[angle_col].to_numpy(dtype=np.float64),
            "P_kistler_gauge_bar":  table[kis_col].to_numpy(dtype=np.float64),
            "P_danfoss_static_bar": table[dan_col].to_numpy(dtype=np.float64),
        })
        return out, fmt

    cols_lower = {h.lower(): h for h in header}
    cycle_col  = next(h for lc, h in cols_lower.items() if "cycle" in lc)
    tdc_col    = next(h for lc, h in cols_lower.items() if "tdc" in lc)
    pulses_col = next(h for lc, h in cols_lower.items() if "pulse" in lc and "cycle" not in lc)
    used = {cycle_col, tdc_col, pulses_col}
    used |= {h for lc, h in cols_lower.items() if "time" in lc or "stamp" in lc}
    pressure_cols = [h for h in header if h not in used]
    kis_col, dan_col = _identify_pressure_columns(table, pressure_cols)

    out = pd.DataFrame({
        "cycle_count":          table[cycle_col].to_numpy(),
        "tdc_marker":           table[tdc_col].to_numpy(),
        "angle_pulses":         table[pulses_col].to_numpy(),
        "P_kistler_gauge_bar":  table[kis_col].to_numpy(dtype=np.float64),
        "P_danfoss_static_bar": table[dan_col].to_numpy(dtype=np.float64),
    })
    return out, fmt


# ──────────────────────────────────────────────────────────────────────
#  4. MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

def parse_file(path: str | Path) -> ParsedFile:
    """Read a unified acquisition file into a structured ParsedFile.

    Reads all three blocks when present. Legacy files with no block headers
    (historical Norbert data) are treated as a bare FAST_DATA table. The
    absolute gas pressure is intentionally NOT computed here; the two raw
    pressure signals are returned separately for the Preprocessor.
    """
    path = Path(path)
    lines = _read_lines_robust(path)

    decimal = _detect_decimal(lines[:30])
    blocks = _split_blocks(lines)

    metadata  = _parse_metadata(blocks.get("METADATA", []))
    slow_data = _parse_slow_data(blocks.get("SLOW_DATA", []), decimal)
    fast_df, fast_fmt = _parse_fast_data(blocks.get("FAST_DATA", []), decimal)

    return ParsedFile(
        metadata=metadata,
        slow_data=slow_data,
        fast_data=fast_df,
        fast_format=fast_fmt,
    )
