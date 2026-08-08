# MM-Routing-MOEA: Reproducibility Guide

This repository contains the completely reproducible codebase for the discrete multimodal routing framework discussed in our research. 

To circumvent GitHub's 100MB file size limit for large network datasets (like `.osm` XMLs and `.graphml` graphs), we employ **Option 1 (Compressed Raw Data Pipeline)**. This guarantees that anyone can clone the repository and automatically regenerate the full 415MB graph deterministically on their local machine without relying on external large file storage.

## 1. System Requirements
- Python 3.10+
- Anaconda / Miniconda (recommended)
- `osmium-tool` (required to decompress and parse the OSM PBF)

### Install OSMIUM
On macOS:
```bash
brew install osmium-tool
```
On Ubuntu/Debian:
```bash
sudo apt-get install osmium-tool
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is not present yet, install the core dependencies: `pip install pymoo networkx pandas numpy osmnx haversine tqdm osmium scikit-learn`)*

---

## 2. Data Preparation (The "Option 1" Method)

The repository explicitly tracks the highly compressed binary files in `data/raw/`:
- `strasbourg.osm.pbf` (~15 MB) : The raw openstreetmap network.
- `strasbourg_gtfs.zip` (~7 MB) : The public transit schedule (bus/tram).

To unpack these and construct the 415 MB discrete multimodal graph (`strasbourg_multimodal.graphml`), simply execute the builder script:

```bash
PYTHONPATH=.:src python src/network/builder.py
```

**What this script does:**
1. It uses `osmium` to rapidly convert the `.pbf` into a `.osm` XML (which is automatically `.gitignore`d).
2. It parses the OSM network via `osmnx` to build the pedestrian layer.
3. It extracts the Bus and Tram routes from the GTFS archive, computing exact transit edges and transfer links.
4. It computes Geodesic (Haversine) lengths for all edges.
5. It exports the final integrated network to `data/processed/strasbourg_multimodal.graphml`.

---

## 3. Running the Discrete MOEA Experiments

With the graph generated, you can launch the fully deterministic experimental pipeline (NSGA-II, Canonical NSGA-III, PI-NSGA-III):

```bash
PYTHONPATH=.:src python src/experiment_runner.py --max-workers 10
```

*Note: The discrete evolutionary algorithms evaluate complex paths and require substantial CPU resources. You can adjust the `--max-workers` parameter to match your machine's CPU core count for optimal parallelization.*

### Outputs
Once completed, the script generates population checkpoints, runtime metrics, and final Hypervolume/IGD indicators in the `outputs_v6_smart` directory.
