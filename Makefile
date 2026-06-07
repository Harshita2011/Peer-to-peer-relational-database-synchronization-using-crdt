.PHONY: test bench install lint clean

install:
	pip install -e ".[dev,bench]"

test:
	python -m pytest tests/ -v --tb=short

test-tombstone:
	python -m pytest tests/test_tombstone.py -v

test-stress:
	python -m pytest tests/test_stress.py -v --timeout=120

test-property:
	python -m pytest tests/test_property.py -v --hypothesis-seed=42

bench:
	python benchmarks/bench_sync_latency.py
	python benchmarks/bench_metadata_growth.py
	python benchmarks/bench_convergence_time.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache build dist *.egg-info
