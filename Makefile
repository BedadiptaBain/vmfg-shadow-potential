PY ?= python3

.PHONY: install test verify data clean
install:
	$(PY) -m pip install -e ".[test]"
test:
	$(PY) -m pytest -q
verify:
	$(PY) scripts/run_all.py
# usage: make data FILE=path/to/DAILY.csv
data:
	$(PY) scripts/06_empirical_section10.py --data $(FILE)
clean:
	rm -rf results/*.md results/*.png .pytest_cache build *.egg-info
