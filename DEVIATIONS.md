# Known Deviations

This file logs any known discrepancies between the outputs of this repository's exact master branch and the numbers printed in the manuscript.

## 1. Network Topology
The manuscript reports the network configuration as **312 nodes and 1847 edges**. 
Running the current reproducible network builder from the frozen OSM snapshot (`data/raw/strasbourg.osm`) and the GTFS archive yields **301 nodes and 1635 edges** (with symmetric road layers disabled and nearest-neighbour $k=1$).

The minor discrepancy arises from updates to the parsing dependencies (`osmium`, GTFS snapshot resolution) and standardizing the topological configuration to ensure strict reproducibility. 

## 2. Floating Point Variations
While the multi-objective optimization algorithms are heavily seeded to be perfectly deterministic across independent parallel runs within the same environment, executing the campaign on different hardware architectures (e.g., Apple Silicon vs. Intel x86) or operating systems may produce minor floating-point divergence in the tail decimals of hypervolume and TwoNN estimates. 
