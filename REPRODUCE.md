# MM-Routing-MOEA: Reproducibility Guide

This repository contains the completely reproducible codebase for the discrete multimodal routing framework discussed in our research. 

To circumvent GitHub's 100MB file size limit for large network datasets, we employ **Option 1 (Compressed Raw Data Pipeline)**. This guarantees that anyone can clone the repository and automatically regenerate the full 415MB graph deterministically on their local machine without relying on external large file storage.

## 1. System Requirements

- **Hardware**: Minimum 8GB RAM (16GB recommended for parallel execution), 4+ CPU Cores
- **OS**: Linux / macOS
- **Python**: 3.9+
- **System Packages**: `osmium-tool` (required to decompress and parse the OSM PBF)

### Install OSMIUM
On macOS:
```bash
brew install osmium-tool
```
On Ubuntu/Debian:
```bash
sudo apt-get install osmium-tool
```

### Environment Setup
Create a virtual environment and install the strictly locked dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate

# For macOS:
pip install -r requirements-macos.lock
# For Linux:
# pip install -r requirements-linux.lock
```

---

## 2. Execute the Reproducibility Pipeline

We provide a strict, fully deterministic pipeline to regenerate the graph, verify correctness, and execute the experiments. **Do not skip the verification step.**

```bash
# 1. Build the Multimodal Graph from compressed OSM/GTFS
PYTHONPATH=.:src python src/network/builder.py

# 2. Run the Strict 4-Phase Verification Pipeline
# This ensures structural validity, baseline accuracy, operator integrity, and MOEA functionality.
PYTHONPATH=.:src python scripts/verify_pipeline.py

# 3. If verification passes (verification_report.json status="SUCCESS"), execute the experiments
PYTHONPATH=.:src python src/experiment_runner.py --max-workers 10
```

### Outputs
Once completed, the experimental scripts generate:
- Population checkpoints (`.csv`) for each generation and algorithm.
- Detailed JSON `run_metadata.json` for provenance.
- Final Hypervolume/IGD indicators in the `outputs_v6_smart` directory.
