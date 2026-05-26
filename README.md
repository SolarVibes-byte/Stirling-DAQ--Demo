# Stirling Cryocooler — Automated DAQ Pipeline

Experimental validation and numerical modelling of a Beta-type Stirling
machine (Sigma 1-125A) operated as a helium cryocooler. This repository
contains the data-acquisition and analysis pipeline developed for the
Bachelor's Thesis (HVL + UPV, 2026).

## What it does

Given a raw acquisition file from the rig, the pipeline:

1. parses the file (historical two-file layout or unified v2.0),
2. reconstructs a clean thermodynamic cycle (angle alignment, ensemble
   averaging, missing-data handling),
3. resolves the boundary temperatures from the slow-data log,
4. evaluates the working-gas volume with the Sage Fourier series and the
   exact crank-slider kinematics,
5. computes the pressure with three models — Experimental (measured),
   Isothermal (Schmidt) and Adiabatic (Urieli) —,
6. computes the refrigeration-mode performance metrics (COP, second-law
   efficiency) by a global energy balance,
7. exports the P-V diagram and the metrics.

## Validation (Case 1)

The reconstructed indicated work is **83.5 J**, within **1 %** of the
published value of 82.6 J (Lümmen & Høeg, 20-ISEC 2024).

## Quick start

Open `demo.ipynb` in Google Colab and run all cells. It clones this
repository, processes the published Case 1 data, and shows the P-V diagram,
the three pressure models and the performance metrics.

## Structure

```
src/      pipeline modules (single-responsibility)
tools/    synthetic dirty-data generator (for stress testing)
data/     Case 1 reference data (Lümmen 2024)
```
