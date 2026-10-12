PYTHON ?= python3
PY := .venv/bin/python
export MPLCONFIGDIR := $(CURDIR)/.venv/mpl

.PHONY: setup test report synthetic criteo

setup: .venv/done
.venv/done: requirements.txt
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -q -r requirements.txt
	touch $@

test: setup
	$(PY) -m pytest -q tests

report: setup
	$(PY) report.py

synthetic: setup
	$(PY) synthetic/pair.py
	$(PY) synthetic/slack.py

criteo: setup
	$(PY) criteo/download.py
	$(PY) criteo/prepare.py
	$(PY) criteo/replay.py
