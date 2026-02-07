# 🧠 AH-RPMNet: Adaptive Hybrid Rock Physics Model Network for Pore Pressure Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of **AH-RPMNet**, a novel hybrid deep learning framework for pore pressure prediction that integrates rock physics modeling with deep neural networks.

## 🎯 Overview

Accurate pore pressure prediction is critical for drilling safety and reservoir evaluation. Traditional methods face limitations: physics-driven approaches rely heavily on empirical trends, while data-driven methods require extensive labeled data. **AH-RPMNet** bridges this gap through a hybrid approach:

1. **Physics-informed pretraining** in normal compaction zones using hydrostatic pressure as labels
2. **Rock physics-enhanced Eaton method** for robust normal compaction velocity calculation
3. **Physics-guided Monte Carlo augmentation** to address data scarcity
4. **Adaptive dual-driven loss** balancing data fidelity and physical consistency

## 📦 Repository Structure

```
AH-RPMNet/
├── 📂 well-log-data/                    # Well log datasets (5 wells)
│   ├── W1.csv, W2.csv, W3.csv, W4.csv, W5.csv
│   └── README_data.md                   # Detailed data description
│
├── 📂 rock-physical-modeling/           # MATLAB rock physics workflow
│   ├── main.m                           # Main execution script
│   ├── SCA.m, dem.m, demyprime.m       # Effective medium theory models
│   ├── gassmink.m, ode45m.m            # Fluid substitution & ODE solvers
│   └── README_matlab.md                 # MATLAB implementation guide
│
├── 📂 pretraining-normal-zones/         # Python pretraining module (PyTorch)
│   ├── example.csv                      # data description (W1)
│   ├── main.py                          # Main training pipeline
│   ├── config.py                        # Hyperparameters & settings
│   ├── model.py                         # CNN-BiGRU network architecture
│   ├── dataloader.py                    # Data loading & preprocessing
│   ├── trainer.py                       # Training loop & optimization
│   ├── evaluator.py                     # Evaluation metrics & analysis
│   └── plot_utils.py                    # Visualization utilities
│
└── 📜 README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

**For MATLAB component:**
- MATLAB R2020a or later
- No additional toolboxes required

**For Python component:**
- Python 3.8 or higher
- PyTorch 1.9+

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yingtian-Liu/AH-RPMNet.git
   cd AH-RPMNet
   ```

2. **Install Python dependencies:**
   ```bash
   cd pretraining-normal-zones
   pip install -r requirements.txt
   ```

3. **Prepare data directories:**
   ```bash
   mkdir -p image invert_checkpoints
   ```

### Usage

#### 1. Rock Physics Modeling (MATLAB)
Execute the multi-stage workflow to compute normal compaction velocity:
```matlab
cd rock-physical-modeling
main.m  % Runs VRH → DEM → SCA → Gassmann workflow
```
Output: `Vn` (normal compaction velocity profile) for each well.

#### 2. Pretraining in Normal Zones (Python)
Train the CNN-BiGRU network using hydrostatic pressure as labels:
```bash
cd pretraining-normal-zones
python main.py
```
The pipeline:
- Loads well log data (density, P-wave velocity, porosity, clay volume)
- Normalizes features using MinMaxScaler
- Trains the hybrid CNN-BiGRU architecture
- Saves model to `invert_checkpoints/pre_train.pth`
- Generates training loss and prediction plots

#### 3. Expected Output
After successful execution, you should see:
```
============================================================
PRETRAINING COMPLETE!
============================================================
Model saved to: ./invert_checkpoints/pre_train.pth
Key metric - RMSE: 1.5315
============================================================
```
And generated plots in the `image/` directory.

## 📊 Data Description

The repository includes data from five wells in the eastern South China Sea:

| Well | Normal Zone Depth (m) | Data Points | Abnormal Zone Depth (m) | Measured Points |
|------|-----------------------|-------------|-------------------------|------------------|
| W1   | 1280–4320             | 28,400      | 4320–4670               | 2                |
| W2   | 1700–3500             | 18,000      | 3500–3900               | 13               |
| W3   | 3290–3770             | 4,800       | 3770–4050               | 12               |
| W4   | 3100–4420             | 13,200      | 4420–4870               | 4                |
| W5   | 4085–4415             | 3,300       | 4415–4915               | 12               |

**Notes:**
- **Normal zone data**: Synthesized using hydrostatic pressure (γₚ = 1)
- **Abnormal zone data**: Actual pressure measurements from MDT/PWD tools
- **Sampling interval**: 0.1 meters for all wells
- **Features included**: Bulk density, P-wave velocity, porosity, clay volume, permeability (log-transformed)

## 🏗️ Model Architecture

### CNN-BiGRU Hybrid Network
The pretraining model combines convolutional and recurrent layers:

```
Input → [Parallel CNN Branches] → Feature Fusion → [Bi-GRU Layers] → Regression
    ├── CNN Branch 1 (dilation=1)    ├── Local pattern extraction
    ├── CNN Branch 2 (dilation=3)    ├── Depth-dependent trend modeling
    └── CNN Branch 3 (dilation=6)    └── Pore pressure coefficient prediction
```

**Key features:**
- **Multi-scale CNN**: Three parallel branches with different dilation rates capture features at various scales
- **Bidirectional GRU**: Models sequential dependencies along depth
- **Group Normalization**: Stabilizes training across different well conditions
- **Residual connections**: Combines CNN and GRU outputs effectively

### Rock Physics Workflow
Four-stage effective medium theory modeling:
1. **VRH Averaging**: Homogenizes brittle minerals (quartz, calcite)
2. **DEM Theory**: Embeds brittle minerals in clay matrix
3. **SCA Model**: Introduces pore structure with calibrated aspect ratios
4. **Gassmann Fluid Substitution**: Saturates rock frame with formation water

## 📈 Performance

AH-RPMNet demonstrates superior performance on South China Sea wells:

| Method | NRMSE | MAPE | Improvement vs. Physics-only |
|--------|-------|------|------------------------------|
| RPM-Eaton (Physics) | 0.067 | 5.73% | Baseline |
| AH-RPMNet (Full) | **0.021** | **1.95%** | **68% NRMSE reduction** |

## ⚠️ Release Notes

### Included in This Release
- ✅ Complete well log data for 5 wells (W1-W5)
- ✅ Full MATLAB implementation of rock physics workflow
- ✅ Modular Python code for pretraining in normal compaction zones
- ✅ Pre-trained model checkpoint
- ✅ Comprehensive documentation and examples

### Not Included (Proprietary Components)
- ❌ Monte Carlo data augmentation module
- ❌ Adaptive dual-driven loss implementation
- ❌ Transfer learning pipeline for overpressure zones
- ❌ Full AH-RPMNet integration code

*These components are part of intellectual property development.*

## 🔍 Key Features

### 1. Physics-Informed Pretraining
- Leverages hydrostatic equilibrium principle: Pₚ = Pw in normal zones
- Generates large-scale synthetic training labels (γₚ = 1)
- Provides physically consistent initialization for transfer learning

### 2. Enhanced Eaton Method
- Replaces empirical NCT with rock physics modeled Vn
- Incorporates mineralogy, porosity, and pore structure
- Reduces subjectivity in trend line fitting

### 3. Modular & Extensible Design
- Clean separation of data, model, training, and evaluation
- Configurable hyperparameters via `config.py`
- Easy integration with existing workflows

For questions, bug reports, or collaboration inquiries, please open an issue on GitHub or contact the authors directly.
---
*This repository is maintained by the Reservoir Geophysics Research Group at Chengdu University of Technology. Last updated: February 2026.*
