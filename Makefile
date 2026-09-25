.PHONY: test estimate e1 e2 e3 help
AWS_PROFILE ?= personal
AWS_REGION ?= us-east-1
export AWS_PROFILE AWS_REGION
MODEL ?= us.anthropic.claude-haiku-4-5-20251001-v1:0

help:
	@grep -E '^[a-z0-9]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -- /'

test:  ## Offline tests. No AWS calls, no cost.
	python3 tests/test_harness.py

estimate:  ## Upper-bound cost of a run, before running it.
	python3 experiments/estimate.py

e1:  ## Tool catalog vs prompt cache (costs money)
	python3 experiments/e1_tool_cache.py --model $(MODEL) $(ARGS)

e2:  ## Effort curve (costs money)
	python3 experiments/e2_effort_curve.py --model $(MODEL) $(ARGS)

e3:  ## Cache-induced output divergence (costs money)
	python3 experiments/e3_cache_divergence.py --model $(MODEL) $(ARGS)
