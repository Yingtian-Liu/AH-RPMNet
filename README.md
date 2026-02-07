# AH-RPMNet: Adaptive Hybrid Rock Physics Model Network for Pore Pressure Prediction

This repository contains the official implementation of **AH-RPMNet**, a hybrid deep learning framework for pore pressure prediction that integrates rock physics modeling with deep neural networks. The method is introduced in the paper:

## 📌 Overview

Accurate pore pressure prediction is crucial for drilling safety and reservoir evaluation. Traditional methods are either purely physics-driven (relying on empirical trends) or purely data-driven (requiring large labeled datasets). AH-RPMNet bridges this gap by:

- **Pretraining in normal compaction zones** using hydrostatic pressure as a physically consistent label.
- **Enhancing the Eaton method** with a multi-stage rock physics workflow (VRH, DEM, SCA, Gassmann) to compute normal compaction velocity.
- **Physics-guided Monte Carlo data augmentation** to overcome data scarcity in overpressured zones.
- **An adaptive dual-driven loss function** that balances data fidelity and physical consistency.

## 📁 Repository Structure

```
.
├── 📂 well-log-data/          # Log data from 5 wells (W1–W5)
│   ├── W1.csv
│   ├── W2.csv
│   ├── W3.csv
│   ├── W4.csv
│   └── W5.csv
├── 📂 rock-physical-modeling/ # MATLAB codes for rock physics modeling
│   ├── main.m                # Main script to run the workflow
│   ├── SCA.m
│   ├── dem.m
│   ├── demyprime.m
│   ├── gassmink.m
│   └── ode45m.m
├── 📂 pretraining-normal-zones/ # PyTorch code for pretraining in normal compaction zones
│   ├── main.py              # Main training script
│   ├── model.py             # Network architecture (CNN + Bi-GRU)
│   ├── dataloader.py        # Data loading and preprocessing
│   └── config.yaml          # Hyperparameters
├── 📜 README.md              # This file
└── 📜 LICENSE                # MIT License
```

## ⚙️ Installation & Dependencies

### Rock Physics Modeling (MATLAB)
- MATLAB R2020a or later
- No additional toolboxes required.

### Pretraining (Python)
- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn, Matplotlib

Install Python dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## 🚀 Quick Start

### 1. Rock Physics Modeling
Run the MATLAB script to compute the normal compaction velocity profile:
```matlab
cd rock-physical-modeling
main.m
```
This executes the multi-stage workflow (VRH → DEM → SCA → Gassmann) and outputs `Vn` (normal compaction velocity).

### 2. Pretraining in Normal Compaction Zones
Prepare the well log data (already provided in `well-log-data/`), then run:
```bash
cd pretraining-normal-zones
python main.py
```
The script will:
- Load log data (density, P-wave velocity, porosity, clay volume)
- Use hydrostatic pressure as label in normal zones
- Train a CNN-BiGRU network
- Save the pretrained model weights

## 📊 Data Description

Five wells from the eastern South China Sea are included:

| Well | Normal Zone Depth (m) | Data Points | Abnormal Zone Depth (m) | Measured Points |
|------|------------------------|-------------|--------------------------|------------------|
| W1   | 1280–4320              | 28400       | 4320–4670                | 2                |
| W2   | 1700–3500              | 18000       | 3500–3900                | 13               |
| W3   | 3290–3770              | 4800        | 3770–4050                | 12               |
| W4   | 3100–4420              | 13200       | 4420–4870                | 4                |
| W5   | 4085–4415              | 3300        | 4415–4915                | 12               |

**Note:**  
- Normal zone data points are synthesized using hydrostatic pressure.
- Abnormal zone points are actual pressure measurements (MDT/PWD).

## 🔧 Key Features of AH-RPMNet

1. **Physics-based pretraining** – Uses hydrostatic equilibrium to generate large-scale training labels.
2. **Rock-physics-enhanced Eaton method** – Replaces empirical NCT with modeled `Vn` from mineralogy and pore geometry.
3. **Monte Carlo augmentation with physical constraints** – Perturbs inputs along regression slopes between logs and Eaton-predicted pressure.
4. **Adaptive dual-driven loss** – Balances data misfit and physics mismatch without manual weighting.
5. **CNN-BiGRU network** – Captures local patterns and depth-dependent trends in log data.

## 📈 Results

AH-RPMNet achieves:
- **NRMSE**: 0.021 (≈68% reduction compared to pure physics-driven methods)
- **MAPE**: 1.95% (≈66% reduction)

## ⚠️ Note on Partial Release

This repository contains:
- ✅ Well log data (5 wells)
- ✅ Rock physics modeling codes (MATLAB)
- ✅ Pretraining codes for normal compaction zones (PyTorch)

Not included in this release:
- ❌ Monte Carlo augmentation module
- ❌ Adaptive loss implementation
- ❌ Full transfer learning pipeline for overpressure zones

These components are withheld due to proprietary augmentation strategies and adaptive loss formulations that are part of ongoing research and intellectual property considerations.

