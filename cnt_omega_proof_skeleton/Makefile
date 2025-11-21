# Makefile
PREREG=docs/prereg_omega.md
CONFIG=configs/omega_srl.yaml

hash-prereg:
	python scripts/hash_file.py $(PREREG) | tee artifacts/prereg_hash.txt

dry-run:
	python -m src.scorecard --config $(CONFIG) --dry_run

prospective:
	python -m src.scorecard --config $(CONFIG) --prospective
