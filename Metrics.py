# Metrics.py
# Refrigeration-mode performance metrics from the global energy balance.
#
# In cooling mode the useful effect is the heat lifted from the cold source
# (the expansion space, at T_source); the cost is the work supplied. The
# machine is treated as a single control volume and the balance is closed with
# the quantities actually measured by the data-acquisition system:
#   - indicated work  W_ind = |contour integral of P dV|  (from the P-V cycle)
#   - heat rejected    Q_out,sink = rho * flow * cp * dT   (from the coolant)
#   - electric power   W_el                                (from the inverter)
#
# The heat lifted from the source is obtained by difference,
#   Q_in,source = Q_out,sink - W_PV,
# which lumps the small environmental leak and axial creep into a single
# unresolved residual. Splitting that residual into its physical components
# requires the analytical loss correlations (Martini 1983, Chahartaghi 2018)
# or the nodal Sage model, and is out of scope here; this is stated explicitly
# so the simplification is transparent.

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import Configuration as Cfg


@dataclass
class PerformanceMetrics:
    """Refrigeration-mode performance summary for one operating point."""
    W_ind_J:            float   # indicated work per cycle (J), |integral P dV|
    W_PV_W:             float   # indicated power (W) = W_ind * rpm / 60
    W_el_W:             float   # electric power drawn (W)
    Q_out_sink_W:       float   # heat rejected to the coolant (W)
    Q_in_source_W:      float   # heat lifted from the cold source (W)
    COP_R_PV:           float   # COP on the indicated (thermodynamic) work
    COP_R_system:       float   # COP on the electric work
    COP_reversible:     float   # Carnot refrigeration COP for these reservoirs
    eta_II_PV:          float   # second-law efficiency on the PV work
    eta_II_system:      float   # second-law efficiency on the electric work
    T_sink_K:           float
    T_source_K:         float


# ──────────────────────────────────────────────────────────────────────
#  1. HEAT REJECTED AT THE SINK (coolant calorimetry)
# ──────────────────────────────────────────────────────────────────────

def compute_heat_rejected(water_flow_Lmin: float,
                          water_delta_T_K: float,
                          ) -> float:
    """Heat rejected to the cooling water, from a simple calorimetric balance.

        Q_out,sink = rho_water * (flow / 60000) * cp_water * dT

    The flow is converted from litres per minute to cubic metres per second
    (divide by 60000); the result is in watts. This is the term validated
    against the reference paper (~2930 W for Case 1).
    """
    mdot_kg_s = Cfg.RHO_WATER_KG_M3 * (water_flow_Lmin / 60000.0)
    return mdot_kg_s * Cfg.CP_WATER_J_KG_K * water_delta_T_K


# ──────────────────────────────────────────────────────────────────────
#  2. FULL PERFORMANCE SET (global balance, cooling mode)
# ──────────────────────────────────────────────────────────────────────

def compute_performance(W_ind_J: float,
                        rpm: float,
                        T_sink_K: float,
                        T_source_K: float,
                        water_flow_Lmin: float,
                        water_delta_T_K: float,
                        power_electric_W: float,
                        ) -> PerformanceMetrics:
    """Compute the refrigeration-mode performance metrics by global balance.

    Parameters come straight from the experimental result and the slow-data
    boundary conditions. The indicated work is taken as a magnitude; its sign
    (cooling vs heating) is a diagnostic handled upstream.

    Returns
    -------
    PerformanceMetrics with the works, heats, COPs and second-law efficiencies.
    """
    W_ind = abs(W_ind_J)

    # Indicated power from the cyclic work and the rotational speed
    W_PV_W = W_ind * rpm / 60.0

    # Heat rejected to the coolant (measured calorimetrically)
    Q_out_sink_W = compute_heat_rejected(water_flow_Lmin, water_delta_T_K)

    # Heat lifted from the cold source, by global balance (the useful effect).
    # The unresolved residual (ambient leak minus axial creep) is absorbed here.
    Q_in_source_W = Q_out_sink_W - W_PV_W

    # Coefficients of performance (refrigeration): useful lift over work cost
    COP_R_PV     = Q_in_source_W / W_PV_W if W_PV_W > 0 else float("nan")
    COP_R_system = Q_in_source_W / power_electric_W if power_electric_W > 0 else float("nan")

    # Reversible (Carnot) refrigeration COP for these reservoir temperatures
    dT = T_sink_K - T_source_K
    COP_reversible = T_source_K / dT if dT > 0 else float("nan")

    # Second-law (exergetic) efficiency: actual COP over reversible COP
    eta_II_PV     = COP_R_PV / COP_reversible if COP_reversible > 0 else float("nan")
    eta_II_system = COP_R_system / COP_reversible if COP_reversible > 0 else float("nan")

    return PerformanceMetrics(
        W_ind_J        = W_ind,
        W_PV_W         = W_PV_W,
        W_el_W         = power_electric_W,
        Q_out_sink_W   = Q_out_sink_W,
        Q_in_source_W  = Q_in_source_W,
        COP_R_PV       = COP_R_PV,
        COP_R_system   = COP_R_system,
        COP_reversible = COP_reversible,
        eta_II_PV      = eta_II_PV,
        eta_II_system  = eta_II_system,
        T_sink_K       = T_sink_K,
        T_source_K     = T_source_K,
    )
