# Visualization.py
# P-V diagram rendering for the automated pipeline.
#
# This module turns a ProcessingResult into a publication-quality P-V figure.
# For the event-driven pipeline it draws the experimental loop only: the
# measured pressure against the Sage Fourier volume, i.e. the validated
# 83 J cycle. It performs no physics; it only reads the result and plots.

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend: render straight to file
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Colour of the experimental loop (near-black: it is the reference curve)
_C_EXP = "#1a1a1a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 1.1,
    "axes.edgecolor": "#333333",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _close_loop(theta_deg: np.ndarray,
               x: np.ndarray,
               y: np.ndarray,
               ) -> tuple[np.ndarray, np.ndarray]:
    """Sort a cyclic loop by angle and repeat the first point so it closes."""
    order = np.argsort(theta_deg)
    xs, ys = x[order], y[order]
    xs = np.append(xs, xs[0])
    ys = np.append(ys, ys[0])
    return xs, ys


def plot_pv_experimental(result, results_dir: str | Path) -> Path:
    """Render the experimental P-V loop (Fourier volume vs measured pressure).

    Only the experimental cycle is drawn: the x-axis is the Sage Fourier total
    volume, the y-axis the reconstructed absolute pressure. The enclosed area
    is the indicated work already reported in the metrics.

    Parameters
    ----------
    result : ProcessingResult
        Output of the orchestrator for one operating point.
    results_dir : path
        Destination directory; created if missing.

    Returns
    -------
    Path to the written PNG.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    V = result.V_fourier_m3 * 1e6                      # m^3 -> cm^3
    Vx, Py = _close_loop(result.theta_deg, V, result.P_experimental_bar)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.fill(Vx, Py, color=_C_EXP, alpha=0.05, zorder=1)
    ax.plot(Vx, Py, color=_C_EXP, lw=2.4, zorder=3,
            label=f"Experimental  (W = {result.W_experimental_J:.1f} J)")

    ax.set_xlabel("Total volume  $V$  [cm³]  (Sage Fourier)")
    ax.set_ylabel("Pressure  $p$  [bar]  (measured)")
    ax.set_title(f"Experimental P–V cycle — {result.case_name}\n"
                 f"{result.T_sink_K:.0f} K sink / {result.T_source_K:.0f} K source "
                 f"· mode: {result.operating_mode}", fontsize=11.5)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#cccccc",
              loc="upper right", fontsize=10.5)
    ax.grid(True, alpha=0.25, lw=0.7)
    ax.margins(x=0.04, y=0.08)

    path = results_dir / f"{result.case_name}_PV.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Rendered P-V diagram %s", path.name)
    return path
