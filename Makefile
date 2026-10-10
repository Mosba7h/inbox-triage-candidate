# Everything runs inside a local virtualenv that is created on first use,
# so `make eval` works on a fresh clone with only LLM_BASE_URL / LLM_MODEL / LLM_API_KEY set.

PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python)
VENV   := .venv
ifeq ($(OS),Windows_NT)
  VPY := $(VENV)/Scripts/python
else
  VPY := $(VENV)/bin/python
endif

.PHONY: install run eval test clean

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --quiet --disable-pip-version-check -r requirements.txt
	@touch $@

install: $(VENV)/.installed

run: install            ## start API + review UI on http://127.0.0.1:8000
	$(VPY) -m triage.api

eval: install           ## triage all 30 tickets with the live model, write eval/results.json
	$(VPY) -m triage.evaluate

test: install           ## unit tests (no model needed)
	$(VPY) -m pytest -q

clean:
	rm -rf $(VENV) data/review_state.json
