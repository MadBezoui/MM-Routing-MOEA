#!/usr/bin/env python3
import os
import urllib.request
import subprocess
import json
import hashlib
from pathlib import Path

DATA_RAW_DIR = Path("data/raw")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

OSM_URL = "https://download.geofabrik.de/europe/france/alsace-latest.osm.pbf"
GTFS_URL = "https://data.strasbourg.eu/api/explore/v2.1/catalog/datasets/horaires-des-lignes-cts-gtfs/exports/zip"

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")

def compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    osm_alsace_path = DATA_RAW_DIR / "alsace-latest.osm.pbf"
    osm_strasbourg_path = DATA_RAW_DIR / "strasbourg.osm.pbf"
    gtfs_path = DATA_RAW_DIR / "strasbourg_gtfs.zip"
    
    # 1. Download
    if not osm_alsace_path.exists():
        download_file(OSM_URL, osm_alsace_path)
    if not gtfs_path.exists():
        download_file(GTFS_URL, gtfs_path)
        
    # 2. Clip OSM data using osmium
    # Strasbourg bounding box: approx 7.67, 48.51, 7.84, 48.65
    print("Clipping Alsace OSM to Strasbourg area using osmium...")
    try:
        subprocess.run([
            "osmium", "extract",
            "-b", "7.67,48.51,7.84,48.65",
            str(osm_alsace_path),
            "-o", str(osm_strasbourg_path),
            "--overwrite"
        ], check=True)
    except FileNotFoundError:
        print("WARNING: osmium-tool is not installed. Using the full Alsace extract.")
        import shutil
        shutil.copy(osm_alsace_path, osm_strasbourg_path)
        
    # 3. Create manifest and checksums
    checksums = {
        "alsace-latest.osm.pbf": compute_sha256(osm_alsace_path),
        "strasbourg.osm.pbf": compute_sha256(osm_strasbourg_path),
        "strasbourg_gtfs.zip": compute_sha256(gtfs_path),
    }
    
    with open(DATA_RAW_DIR / "checksums.sha256", "w") as f:
        for k, v in checksums.items():
            f.write(f"{v}  {k}\n")
            
    with open(DATA_RAW_DIR / "sources.json", "w") as f:
        json.dump({
            "osm": OSM_URL,
            "gtfs": GTFS_URL,
            "bounding_box": "7.67,48.51,7.84,48.65"
        }, f, indent=2)
        
    print("Network data preparation complete.")

if __name__ == "__main__":
    main()
