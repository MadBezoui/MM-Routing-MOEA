.PHONY: data network smoke reproduce

data:
	docker build -f Dockerfile.data -t mm-routing-data .
	docker run --rm -v "$(PWD):/workspace" mm-routing-data

network:
	python -m src.network.builder
	python -m src.network.descriptors --out results/network

smoke:
	python -m experiments.smoke_test

reproduce:
	./reproduce.sh --all
