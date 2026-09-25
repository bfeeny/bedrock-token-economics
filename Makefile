.PHONY: test estimate e1 e2 e3 venv help
AWS_PROFILE ?= personal
AWS_REGION ?= us-east-1
export AWS_PROFILE AWS_REGION
MODEL ?= us.anthropic.claude-haiku-4-5-20251001-v1:0
# The system interpreter's botocore predates cachePoint and outputConfig.effort,
# so the harness runs in its own venv. `make venv` creates it.
PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

help:
	@grep -E '^[a-z0-9]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -- /'

test:  ## Offline tests. No AWS calls, no cost.
	$(PY) tests/test_harness.py

estimate:  ## Upper-bound cost of a run, before running it.
	$(PY) experiments/estimate.py

e1:  ## Tool catalog vs prompt cache (costs money)
	$(PY) experiments/e1_tool_cache.py --model $(MODEL) $(ARGS)

e2:  ## Effort curve (costs money)
	$(PY) experiments/e2_effort_curve.py --model $(MODEL) $(ARGS)

e3:  ## Cache-induced output divergence (costs money)
	$(PY) experiments/e3_cache_divergence.py --model $(MODEL) $(ARGS)

venv:  ## Create the venv the live experiments need
	python3 -m venv .venv && .venv/bin/pip install -q --upgrade -r requirements.txt
	@echo "venv ready: $$(.venv/bin/python -c 'import botocore;print(botocore.__version__)')"
