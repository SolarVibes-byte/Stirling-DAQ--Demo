# make_dummy_data.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import Configuration as Cfg
import Parsers as Prs


def build_clean_template(seed_path: Path):
    parsed = Prs.parse_file(seed_path)
    fast = parsed.fast_data
    theta = np.mod(fast["theta_deg"].to_numpy(), 360.0)
    P_kis = fast["P_kistler_gauge_bar"].to_numpy()
    P_dan = fast["P_danfoss_static_bar"].to_numpy()
    order = np.argsort(theta)
    theta_s, P_kis_s, P_dan_s = theta[order], P_kis[order], P_dan[order]
    n_grid = Cfg.ENCODER_PULSES_PER_REV
    theta_grid = np.arange(n_grid) * Cfg.ENCODER_DEG_PER_PULSE
    P_kis_tpl = np.interp(theta_grid, theta_s, P_kis_s, period=360.0)
    P_dan_tpl = np.interp(theta_grid, theta_s, P_dan_s, period=360.0)
    return theta_grid, P_kis_tpl, P_dan_tpl


def make_dummy(seed_path, out_path, n_cycles=Cfg.N_CYCLES_PER_RECORDING,
               n_transient=Cfg.N_CYCLES_DISCARDED_INIT, noise_sigma_bar=0.1,
               cycle_variation_bar=0.05, transient_drift_bar=1.5,
               encoder_offset_deg=0.0, random_start_deg=0.0, rpm=1400.0, nan_probability=0.0, seed=42):
    rng = np.random.default_rng(seed)
    theta_grid, P_kis_tpl, P_dan_tpl = build_clean_template(seed_path)
    n_grid = len(theta_grid)
    deg_per_pulse = Cfg.ENCODER_DEG_PER_PULSE

    def kis_at(angle):
        return np.interp(np.mod(angle, 360.0), theta_grid, P_kis_tpl, period=360.0)
    def dan_at(angle):
        return np.interp(np.mod(angle, 360.0), theta_grid, P_dan_tpl, period=360.0)

    max_start_pulse = int(round(random_start_deg / deg_per_pulse))
    start_pulse = int(rng.integers(-max_start_pulse, max_start_pulse + 1)) if max_start_pulse > 0 else 0
    n_record = (n_cycles + 1) * n_grid
    cycle_count = np.zeros(n_record, dtype=int)
    tdc_marker = np.zeros(n_record, dtype=int)
    angle_pulses = np.arange(n_record, dtype=int)
    P_kis = np.zeros(n_record); P_dan = np.zeros(n_record)
    markers_seen = 0
    for j in range(n_record):
        abs_pulse = start_pulse + j
        within = abs_pulse % n_grid
        if within == 0:
            tdc_marker[j] = 1; markers_seen += 1
        cycle_count[j] = markers_seen
        enc_angle = within * deg_per_pulse
        true_angle = enc_angle - encoder_offset_deg
        kis_val = float(kis_at(np.array([true_angle]))[0])
        dan_val = float(dan_at(np.array([true_angle]))[0])
        rec_cycle = markers_seen
        if 1 <= rec_cycle <= n_transient and transient_drift_bar > 0.0:
            kis_val += (n_transient - rec_cycle + 1) / n_transient * transient_drift_bar
        kis_val += rng.normal(0.0, cycle_variation_bar)
        P_kis[j] = kis_val; P_dan[j] = dan_val
    P_kis += rng.normal(0.0, noise_sigma_bar, size=n_record)
    if nan_probability > 0.0:
        mask = rng.random(n_record) < nan_probability
        P_kis[mask] = np.nan
    dt = 60.0 / (rpm * n_grid)
    timestamp = angle_pulses * dt
    metadata = {"File_Version": "2.0", "Mode": "cooling", "Charge_Pressure_bar": "21.0",
                "RPM_setpoint": "1400", "Encoder_Pulses_Per_Rev": str(n_grid),
                "Revolutions": str(n_cycles), "Pressure_Format": "gauge_raw",
                "Encoder_Offset_deg": str(encoder_offset_deg), "Source": f"synthetic from {seed_path.name}"}
    slow_data = {"T_cold_head_C": "-114.5", "T_cooling_water_in_C": "11.8",
                 "T_cooling_water_out_C": "16.6", "RPM_measured": "-1380.4",
                 "Torque_pct": "-126.0", "Power_electric_W": "3690.0", "Cooling_water_flow_Lmin": "9.2"}
    lines = ["[METADATA]"]
    lines += [f"{k};{v}" for k, v in metadata.items()]
    lines.append("[SLOW_DATA]")
    lines += [f"{k};{v}" for k, v in slow_data.items()]
    lines.append("[FAST_DATA]")
    lines.append("Cycle_Count;DI_TDC_Marker;DI_Angle_Pulses;Timestamp;P_danfoss_static_bar;P_kistler_gauge_bar")
    for j in range(n_record):
        lines.append(f"{cycle_count[j]};{tdc_marker[j]};{angle_pulses[j]};{timestamp[j]:.6f};{P_dan[j]:.4f};{P_kis[j]:.4f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tpl_path = out_path.with_name(out_path.stem + "_template.csv")
    pd.DataFrame({"theta_deg": theta_grid, "P_kistler_gauge_bar": P_kis_tpl,
                  "P_danfoss_static_bar": P_dan_tpl}).to_csv(tpl_path, index=False)


if __name__ == "__main__":
    seed_file = PROJECT_ROOT / "data" / "case1.txt"
    make_dummy(seed_file, PROJECT_ROOT / "data" / "case1_dirty.txt",
               encoder_offset_deg=Cfg.ENCODER_OFFSET_DEG, random_start_deg=45.0)
    print("Dirty file (level 5) written.")
