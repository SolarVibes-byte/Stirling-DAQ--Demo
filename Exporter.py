# Exporter.py
# Clean, traceable export of processing results to CSV.
#
# Each processed operating point produces two CSV files in the results
# directory, named after the case:
#   <case>_curves.csv   — per-angle curves (theta, volumes, pressures) for
#                         regenerating the P-V diagram downstream.
#   <case>_metrics.csv  — single-row scalar summary (works, temperatures,
#                         mode) for tabulating across many runs.
#
# The module only formats and writes; it performs no physics.

from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def export_result(result, results_dir: str | Path) -> tuple[Path, Path]:
    """Write the per-angle curves and the scalar metrics of one result.

    Parameters
    ----------
    result : ProcessingResult
        The orchestrator output for one operating point.
    results_dir : path
        Destination directory; created if missing.

    Returns
    -------
    (curves_path, metrics_path) : the two CSV files written.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    case = result.case_name

    curves = pd.DataFrame({
        "theta_deg":            result.theta_deg,
        "V_fourier_cm3":        result.V_fourier_m3 * 1e6,
        "V_crank_slider_cm3":   result.V_crank_slider_m3 * 1e6,
        "P_experimental_bar":   result.P_experimental_bar,
        "P_schmidt_bar":        result.P_schmidt_bar,
        "P_adiabatic_bar":      result.P_adiabatic_bar,
    })
    curves_path = results_dir / f"{case}_curves.csv"
    curves.to_csv(curves_path, index=False)

    row = {
        "case":              case,
        "operating_mode":    result.operating_mode,
        "W_experimental_J":  round(result.W_experimental_J, 3),
        "W_schmidt_J":       round(result.W_schmidt_J, 3),
        "W_adiabatic_J":     round(result.W_adiabatic_J, 3),
        "T_sink_K":          round(result.T_sink_K, 2),
        "T_source_K":        round(result.T_source_K, 2),
    }
    # Append the refrigeration-mode KPIs when the balance could be closed.
    if result.metrics is not None:
        m = result.metrics
        row.update({
            "W_PV_W":         round(m.W_PV_W, 1),
            "W_el_W":         round(m.W_el_W, 1),
            "Q_out_sink_W":   round(m.Q_out_sink_W, 1),
            "Q_in_source_W":  round(m.Q_in_source_W, 1),
            "COP_R_PV":       round(m.COP_R_PV, 4),
            "COP_R_system":   round(m.COP_R_system, 4),
            "COP_reversible": round(m.COP_reversible, 4),
            "eta_II_PV":      round(m.eta_II_PV, 4),
            "eta_II_system":  round(m.eta_II_system, 4),
        })
    metrics = pd.DataFrame([row])
    metrics_path = results_dir / f"{case}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    logger.info("Exported %s and %s", curves_path.name, metrics_path.name)
    return curves_path, metrics_path
