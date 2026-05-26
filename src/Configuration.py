# Configuration.py
# Single source of truth for the Sigma 1-125A Beta-Stirling cryocooler.
# All physical, geometric, and calibration constants reside here and are
# imported by every other module of the ecosystem.

from __future__ import annotations
from enum import Enum
from typing import Final
import math


# ── Angular conventions ──
class TDCReference(Enum):
    PISTON    = "piston"      # theta = 0 at power-piston TDC (ideal models)
    DISPLACER = "displacer"   # theta = 0 at displacer TDC (experimental data)

TDC_REFERENCE_IDEAL:        Final[TDCReference] = TDCReference.PISTON
TDC_REFERENCE_EXPERIMENTAL: Final[TDCReference] = TDCReference.DISPLACER

PHASE_OFFSET_DEG: Final[float] = 60.0
PHASE_OFFSET_RAD: Final[float] = math.radians(PHASE_OFFSET_DEG)


# ── Engine geometry (Lummen2024 Table 1) ──
CYLINDER_DIAMETER_M:        Final[float] = 65.0e-3
DISPLACER_CRANK_RADIUS_M:   Final[float] = 19.05e-3
DISPLACER_ROD_LENGTH_M:     Final[float] = 85.2e-3
PISTON_CRANK_RADIUS_M:      Final[float] = 19.0e-3
PISTON_ROD_LENGTH_M:        Final[float] = 175.0e-3
NOMINAL_RPM:                Final[int]   = 1400


# ── Engine volumes (April 2026 revision) ──
V_SWEPT_COMPRESSION_M3:     Final[float] = 123.12e-6
V_SWEPT_EXPANSION_M3:       Final[float] = 126.40e-6
V_COOLER_HX_M3:             Final[float] = 13.73e-6
V_DEAD_COOLER_M3:           Final[float] = 22.70e-6
V_REGEN_MATRIX_62_M3:       Final[float] = 110.00e-6
V_REGEN_MATRIX_68_M3:       Final[float] = 120.60e-6
V_DEAD_REGEN_M3:            Final[float] = 12.20e-6
V_HEATER_HX_HEAD_1_M3:      Final[float] = 28.27e-6
V_HEATER_HX_HEAD_3_M3:      Final[float] = 39.71e-6

V_CONST_TABLE: Final[dict[tuple[int, int], float]] = {
    (1, 62): 186.90e-6, (1, 68): 197.50e-6,
    (3, 62): 198.34e-6, (3, 68): 208.94e-6,
}


# ── Working fluid: helium ──
R_HELIUM_J_KG_K:    Final[float] = 2077.0
GAMMA_HELIUM:       Final[float] = 5.0 / 3.0


# ── Experimental validation cases (Lummen2024 Table 4) ──
CASE_1: Final[dict[str, float]] = {
    "pressure_bar": 21.0, "rpm": 1400.0, "T_sink_K": 285.0, "T_source_K": 159.0,
    "W_PV_W": 1927.0, "W_el_W": 3650.0, "Q_in_source_W": 939.0,
    "Q_in_ambient_W": 208.0, "Q_out_sink_W": 2930.0, "Q_out_creep_W": 202.0,
    "W_ind_J_per_cycle": 82.6, "eta_el_crank": 0.528,
    "COP_R_PV": 0.473, "COP_R_system": 0.250,
}
CASE_2: Final[dict[str, float]] = {
    "pressure_bar": 18.5, "rpm": 1308.0, "T_sink_K": 279.0, "T_source_K": 153.0,
    "W_PV_W": 1670.0, "W_el_W": 3279.0, "Q_in_source_W": 742.0,
    "Q_in_ambient_W": 229.0, "Q_out_sink_W": 2419.0, "Q_out_creep_W": 166.0,
    "W_ind_J_per_cycle": 76.6, "eta_el_crank": 0.509,
    "COP_R_PV": 0.460, "COP_R_system": 0.234,
}


# ── Electromechanical chain (Lonne motor + ABB inverter) ──
ETA_INVERTER_NOMINAL:   Final[float] = 0.955
ETA_MOTOR_NOMINAL:      Final[float] = 0.896
ETA_ELECTRICAL_NOMINAL: Final[float] = ETA_INVERTER_NOMINAL * ETA_MOTOR_NOMINAL
ETA_MECH_NOMINAL:       Final[float] = 0.617


# ── Sensor and encoder calibration ──
ENCODER_OFFSET_DEG:       Final[float] = 1.3
ENCODER_PULSES_PER_REV:   Final[int]   = 120
ENCODER_DEG_PER_PULSE:    Final[float] = 360.0 / ENCODER_PULSES_PER_REV
ENCODER_SAMPLING_DEG:     Final[float] = 3.0
ANGULAR_RESOLUTION_DEG:   Final[float] = 0.5
ANGULAR_POINTS_PER_CYCLE: Final[int]   = int(360.0 / ANGULAR_RESOLUTION_DEG)

# Standard atmospheric reference used to convert the gauge-referenced sensor
# reading into absolute gas pressure. Configurable for tests where the actual
# barometric pressure of the test day is known.
ATMOSPHERIC_PRESSURE_BAR: Final[float] = 1.013

# Cooling-water thermophysical properties at the mean coolant temperature
# (~14 C), used to convert the measured flow and temperature rise into the
# heat rejected at the sink: Q_out = rho * (flow/60000) * cp * dT.
RHO_WATER_KG_M3:    Final[float] = 999.0      # kg/m^3
CP_WATER_J_KG_K:    Final[float] = 4186.0     # J/(kg.K)

# Laboratory ambient temperature (used only as a reference for reporting the
# net environmental term; the balance itself does not require it).
T_AMBIENT_K:        Final[float] = 298.0      # K


# ── Ensemble averaging (data acquisition spec) ──
N_CYCLES_PER_RECORDING:  Final[int] = 10
N_CYCLES_DISCARDED_INIT: Final[int] = 2

# ── Missing-data handling (stress-test robustness) ──
# A cycle with a missing-sample fraction above this threshold is discarded
# rather than interpolated; below it, isolated gaps are linearly interpolated.
MAX_NAN_FRACTION_PER_CYCLE: Final[float] = 0.30
# Minimum number of clean cycles required to trust the ensemble average;
# below this the recording is rejected.
MIN_CLEAN_CYCLES:           Final[int]   = 3


# ── Helper accessors ──
def get_total_fixed_volume(head: int, porosity_percent: int) -> float:
    key = (head, porosity_percent)
    if key not in V_CONST_TABLE:
        raise ValueError(f"Unknown configuration: head={head}, porosity={porosity_percent}")
    return V_CONST_TABLE[key]

def get_regenerator_volume(porosity_percent: int) -> float:
    if porosity_percent == 62:
        return V_REGEN_MATRIX_62_M3
    elif porosity_percent == 68:
        return V_REGEN_MATRIX_68_M3
    raise ValueError(f"Unsupported porosity: {porosity_percent}")

def get_heater_volume(head: int) -> float:
    if head == 1:
        return V_HEATER_HX_HEAD_1_M3
    elif head == 3:
        return V_HEATER_HX_HEAD_3_M3
    raise ValueError(f"Unsupported head: {head}")
