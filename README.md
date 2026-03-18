# JaGuard: Jamming Correction of GNSS Deviation with Deep Temporal Graphs

## Overview

**JaGuard (Jamming Guardian)** is the first deep temporal graph neural network designed to estimate and correct jamming-induced positional drift in GNSS systems. 
JaGuard reformulates mitigation as a **dynamic graph regression problem**. It models the satellite-receiver constellation as a sequence of heterogeneous star graphs, capturing the physical deterioration of the signal over time.

### Key Features:
**Dynamic Star Graph:** Models the receiver as a central node and visible satellites as leaf nodes.
**Spatiotemporal Fusion:** Uses a **HeteroGCLSTM** layer to process 10-second windows of signal history.
**Minimalist Input:** Operates exclusively on standard NMEA observables (SNR, Azimuth, Elevation; Latitude and Longitude).
**High Resilience:** Maintains centimeter-level accuracy even under severe -45 dBm jamming and  data starvation.


## Project Structure

```text
├── gnss/                  # Core library
│   ├── train/             # Training logic and LSTM gate definitions
│   ├── dataset.py         # Graph construction, sliding windows & normalization
│   └── model.py           # JaGuard architecture (HeteroGCLSTM)
├── params.yaml            # Central experiment configuration
├── prepare_data.py        # Data preprocessing (NMEA → Z-score normalized graphs)
├── run_experiment.py      # Execution for a single configuration/seed
├── run_all_experiments.py # Master script for automated experimental sweeps
├── dvc.yaml               # DVC pipeline orchestration
└── README.md 
```
## Installation

Make sure you have [Conda](https://docs.conda.io/en/latest/) installed:

### 1. Create environment
conda create --solver classic -n gnss-py310 \
  python=3.10 \
  numpy=1.24.4 \
  scipy=1.15.2 \
  pandas=1.3.5 \
  scikit-learn \
  -c conda-forge -y

### 2. Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gnss-py310

### 3. Install remaining dependencies
pip install -r requirements.txt


## Automated Pipeline

This project is fully instrumented with Data Version Control (DVC) to ensure reproducibility. To simplify the research workflow, we use an automated sweep script that manages parameter updates and triggers the DVC pipeline internally. 
To automate the evaluation across all discovered receivers, jamming types, and power levels, use the run_all_experiments.py script. This script automatically updates params.yaml for each configuration and executes dvc repro for you.
### 1. Run the full sweep with default settings 
python run_all_experiments.py

### 2. Optional: Run a dry-run to see the experiment matrix without executing
python run_all_experiments.py --dry-run

#### 3. Optional: Filter by specific receivers or define custom seeds
python run_all_experiments.py --receivers Ublox10,GP01 --seeds 42,2024

## Citation




