# ThermodynamicsSchmidt.py — first-order isothermal model
from __future__ import annotations
import numpy as np
import Configuration as Cfg

def compute_pressure(V_c, V_e, T_compression_K, T_expansion_K, p_mean_Pa,
                     head=1, porosity_percent=62):
    V_cooler = Cfg.V_COOLER_HX_M3
    V_heater = Cfg.get_heater_volume(head)
    V_regen = Cfg.get_regenerator_volume(porosity_percent)
    T_k = T_compression_K; T_h = T_expansion_K
    T_r = (T_h - T_k) / np.log(T_h / T_k)
    S = (V_c / T_k + V_e / T_h + V_cooler / T_k + V_heater / T_h + V_regen / T_r)
    inv_S_mean = np.mean(1.0 / S)
    mR = p_mean_Pa / inv_S_mean
    return mR / S

def compute_indicated_work(P, V_total):
    P_closed = np.append(P, P[0])
    V_closed = np.append(V_total, V_total[0])
    return abs(np.trapezoid(P_closed, V_closed))

def compute_indicated_power(W_ind_J, rpm):
    return W_ind_J * rpm / 60.0
