#!/usr/bin/env python3
"""
Download and prepare the frozen OSM and GTFS inputs used by the PI-NSGA-III multimodal transportation experiments.
The script is intentionally zero-configuration. It downloads:
1. The frozen Geofabrik Alsace OSM snapshot dated 2026-01-01.
2. The Eurometropole de Strasbourg administrative boundary.
3. The archived CTS GTFS feed published on 2026-08-05.
4. A Strasbourg OSM extract clipped with complete ways.
5. Checksums, provenance metadata and experiment context.
Run from the repository root:
    python scripts/download_network_data.py
System dependency: osmium-tool
A Dockerfile is provided to avoid manual installation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORY = REPOSITORY_ROOT / "data" / "raw"

# Frozen OSM snapshot: 1 January 2026.
OSM_SOURCE_URL = (
    "https://download.geofabrik.de/europe/france/"
    "alsace-260101.osm.pbf"
)
OSM_MD5_URL = (
    "https://download.geofabrik.de/europe/france/"
    "alsace-260101.osm.pbf.md5"
)

# Eurometropole de Strasbourg, official EPCI code 246700488.
BOUNDARY_URL = (
    "https://geo.api.gouv.fr/epcis/246700488"
    "?format=geojson&geometry=contour"
)

# Frozen CTS GTFS archive published on 5 August 2026.
GTFS_SOURCE_URL = (
    "https://transport-data-gouv-fr-resource-history-prod."
    "cellar-c2.services.clever-cloud.com/79220/"
    "79220.20260805.121649.272855.zip"
)

USER_AGENT = (
    "PI-NSGA-III-Multimodal-Transportation/"
    "1.0 (scientific reproducibility)"
)

REFERENCE_SERVICE_DATE = "2026-09-15"
REFERENCE_DEPARTURE_TIME = "08:00:00"
REFERENCE_TIMEZONE = "Europe/Paris"

REQUIRED_GTFS_FILES = {
    "agency.txt",
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
}

def digest(path: Path, algorithm: str) -> str:
    """Return a hexadecimal checksum for a file."""
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()

def download(url: str, destination: Path) -> None:
    """Download a URL atomically."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[reuse] {destination.relative_to(REPOSITORY_ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    print(f"[download] {url}")
    with urllib.request.urlopen(request, timeout=300) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output)

    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {url}")

    temporary.replace(destination)
    print(
        f"[saved] {destination.relative_to(REPOSITORY_ROOT)} "
        f"({destination.stat().st_size:,} bytes)"
    )

def read_expected_md5(path: Path) -> str:
    """Extract the MD5 value from a Geofabrik checksum file."""
    content = path.read_text(encoding="utf-8").strip()
    value = content.split()[0].lower()
    if len(value) != 32:
        raise RuntimeError(
            f"Invalid Geofabrik MD5 file: {content!r}"
        )
    return value

def validate_osm_source(
    osm_path: Path,
    md5_path: Path,
) -> None:
    """Validate the downloaded Geofabrik snapshot."""
    expected = read_expected_md5(md5_path)
    observed = digest(osm_path, "md5")
    if expected != observed:
        raise RuntimeError(
            "OSM checksum mismatch:\n"
            f"expected: {expected}\n"
            f"observed: {observed}"
        )
    print(f"[validated] OSM MD5 {observed}")

def validate_boundary(path: Path) -> None:
    """Validate the downloaded GeoJSON boundary."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "Feature":
        raise RuntimeError("Boundary is not a GeoJSON Feature.")
    properties = data.get("properties", {})
    geometry = data.get("geometry", {})
    if properties.get("code") != "246700488":
        raise RuntimeError(
            "Downloaded boundary does not correspond to "
            "EPCI 246700488."
        )
    if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise RuntimeError("Unexpected boundary geometry type.")
    print(
        "[validated] Eurometropole de Strasbourg boundary, "
        "EPCI 246700488"
    )

def validate_gtfs(path: Path) -> list[str]:
    """Validate the basic structure of the CTS GTFS archive."""
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Invalid ZIP file: {path}")

    with zipfile.ZipFile(path) as archive:
        names = {
            Path(name).name
            for name in archive.namelist()
            if not name.endswith("/")
        }
        missing = REQUIRED_GTFS_FILES - names
        if missing:
            raise RuntimeError(
                "The GTFS archive is incomplete. Missing files: "
                + ", ".join(sorted(missing))
            )
        bad_members = archive.testzip()
        if bad_members is not None:
            raise RuntimeError(
                f"Corrupted GTFS member: {bad_members}"
            )
    print(
        "[validated] CTS GTFS archive: "
        f"{len(names)} files, required tables present"
    )
    return sorted(names)

def require_osmium() -> str:
    """Return the osmium executable or stop with a clear error."""
    executable = shutil.which("osmium")
    if executable is None:
        raise RuntimeError(
            "The 'osmium' executable is unavailable.\n"
            "Use the supplied Docker command:\n\n"
            "docker build -t pi-nsga3-data .\n"
            "docker run --rm -v \"${PWD}:/workspace\" "
            "pi-nsga3-data\n"
        )
    return executable

def create_strasbourg_extract(
    osmium: str,
    source: Path,
    boundary: Path,
    destination: Path,
) -> None:
    """Clip the Alsace snapshot to the Strasbourg EPCI boundary."""
    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"[reuse] "
            f"{destination.relative_to(REPOSITORY_ROOT)}"
        )
        return

    command = [
        osmium,
        "extract",
        "--polygon",
        str(boundary),
        "--strategy",
        "complete_ways",
        "--overwrite",
        "--output",
        str(destination),
        str(source),
    ]

    print("[execute] " + " ".join(command))
    subprocess.run(command, check=True)

    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError("OSM extraction produced no output.")

    print(
        f"[created] {destination.relative_to(REPOSITORY_ROOT)} "
        f"({destination.stat().st_size:,} bytes)"
    )

def osm_file_information(
    osmium: str,
    path: Path,
) -> dict:
    """Collect metadata reported by osmium fileinfo."""
    process = subprocess.run(
        [
            osmium,
            "fileinfo",
            "--extended",
            "--json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout)

def write_checksums(paths: list[Path]) -> None:
    """Write SHA-256 checksums for every generated input."""
    checksum_path = RAW_DIRECTORY / "checksums.sha256"
    with checksum_path.open("w", encoding="utf-8") as stream:
        for path in paths:
            relative_name = path.relative_to(RAW_DIRECTORY)
            stream.write(
                f"{digest(path, 'sha256')}  {relative_name}\n"
            )
    print(
        f"[created] "
        f"{checksum_path.relative_to(REPOSITORY_ROOT)}"
    )

def main() -> int:
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)

    alsace_path = RAW_DIRECTORY / "alsace-260101.osm.pbf"
    osm_md5_path = RAW_DIRECTORY / "alsace-260101.osm.pbf.md5"
    boundary_path = (
        RAW_DIRECTORY / "eurometropole_strasbourg_246700488.geojson"
    )
    strasbourg_path = RAW_DIRECTORY / "strasbourg.osm"
    gtfs_path = RAW_DIRECTORY / "strasbourg_gtfs.zip"

    download(OSM_SOURCE_URL, alsace_path)
    download(OSM_MD5_URL, osm_md5_path)
    download(BOUNDARY_URL, boundary_path)
    download(GTFS_SOURCE_URL, gtfs_path)

    validate_osm_source(alsace_path, osm_md5_path)
    validate_boundary(boundary_path)
    gtfs_members = validate_gtfs(gtfs_path)

    osmium = require_osmium()
    create_strasbourg_extract(
        osmium=osmium,
        source=alsace_path,
        boundary=boundary_path,
        destination=strasbourg_path,
    )
    osm_information = osm_file_information(
        osmium,
        strasbourg_path,
    )

    tracked_files = [
        alsace_path,
        osm_md5_path,
        boundary_path,
        strasbourg_path,
        gtfs_path,
    ]
    write_checksums(tracked_files)

    provenance = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "study_area": {
            "name": "Eurometropole de Strasbourg",
            "epci_code": "246700488",
            "boundary_source": BOUNDARY_URL,
            "boundary_file": boundary_path.name,
            "clipping_strategy": "complete_ways",
        },
        "osm": {
            "provider": "Geofabrik",
            "data_source": "OpenStreetMap",
            "snapshot": "2026-01-01",
            "source_url": OSM_SOURCE_URL,
            "source_file": alsace_path.name,
            "source_md5": digest(alsace_path, "md5"),
            "source_sha256": digest(
                alsace_path,
                "sha256",
            ),
            "extract_file": strasbourg_path.name,
            "extract_sha256": digest(
                strasbourg_path,
                "sha256",
            ),
            "osmium_fileinfo": osm_information,
        },
        "gtfs": {
            "operator": (
                "Compagnie des Transports Strasbourgeois"
            ),
            "network": "CTS",
            "publication_date": "2026-08-05",
            "valid_from": "2026-08-05",
            "valid_until": "2027-02-01",
            "source_url": GTFS_SOURCE_URL,
            "file": gtfs_path.name,
            "sha256": digest(gtfs_path, "sha256"),
            "members": gtfs_members,
        },
        "experiment_time": {
            "service_date": REFERENCE_SERVICE_DATE,
            "departure_time": REFERENCE_DEPARTURE_TIME,
            "timezone": REFERENCE_TIMEZONE,
        },
        "licenses": {
            "openstreetmap": (
                "Open Database License (ODbL) 1.0"
            ),
            "gtfs": (
                "Licence stated by the French National "
                "Access Point resource page"
            ),
        },
    }

    provenance_path = RAW_DIRECTORY / "sources.json"
    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    context = {
        "study_area": "Eurometropole de Strasbourg",
        "epci_code": "246700488",
        "osm_file": "data/raw/strasbourg.osm",
        "gtfs_file": "data/raw/strasbourg_gtfs.zip",
        "service_date": REFERENCE_SERVICE_DATE,
        "departure_time": REFERENCE_DEPARTURE_TIME,
        "timezone": REFERENCE_TIMEZONE,
    }

    context_path = (
        RAW_DIRECTORY / "experiment_context.json"
    )
    context_path.write_text(
        json.dumps(context, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("[success] Network data preparation completed.")
    print(f"OSM: {strasbourg_path}")
    print(f"GTFS: {gtfs_path}")
    print(f"Date: {REFERENCE_SERVICE_DATE}")
    print(f"Time: {REFERENCE_DEPARTURE_TIME}")
    print(f"Zone: {REFERENCE_TIMEZONE}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)
