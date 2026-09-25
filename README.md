# Token economics on Amazon Bedrock

A measurement harness for techniques that claim to reduce LLM token consumption,
run against Bedrock and reported in the two currencies a Bedrock customer
actually pays in.

## The method, stated once

Four rules, each of which exists because the published literature routinely
breaks it.

**1. Two currencies, always both.** A Bedrock call spends dollars *and* quota,
and they are not proportional. Quota consumption is
`input + cacheWrite + (output x burndown)`, where burndown is 5x to 15x for
Anthropic models and 1:1 elsewhere. Cache *reads* cost dollars at a 90% discount
and cost **nothing at all** against quota. So a technique that trades input
tokens for output tokens can improve the token count and worsen the bill, and a
technique that breaks a cached prefix can improve the token count and worsen
both. Every result here reports dollars and burndown-weighted quota separately.

**2. Never blend the token counts.** Bedrock reports input, output, cache-read
and cache-write separately because they are priced separately. This harness
never adds them up.

**3. Assert that the mechanism engaged.** A cache checkpoint below the model's
minimum prefix is ignored and the request still succeeds — silently. Haiku 4.5
needs 4,096 tokens where the Opus 5 family needs 512. Any arm that expects
caching and never observes a cache read or write is reported as a broken run,
not as a null result.

**4. Distributions, not means.** Run-to-run token spend on an identical agentic
task has been measured varying by up to 30x. Percentiles and a paired bootstrap
on the difference; a mean of three runs is noise.

And one editorial rule: every published number gets the flattering task it came
from named next to it.

## What is here

| | |
|---|---|
| `tokens/prices.py` | Prices and burndown, with provenance per entry. Unverified rates are `None` — the dollar column is omitted rather than guessed. |
| `tokens/meter.py` | One call, fully accounted, appended to JSONL. Analysis never re-runs inference. |
| `tokens/stats.py` | Percentiles and paired bootstrap intervals. |
| `tokens/workloads.py` | Tool catalogs and prefixes. Tool *count* and schema *verbosity* vary independently, because conflating them is the standard error in this literature. |
| `experiments/e1_tool_cache.py` | Does retrieving tools cost more than caching them? |
| `experiments/e2_effort_curve.py` | What does `effort` cost and buy? |
| `experiments/e3_cache_divergence.py` | Does a cache hit change the answer? |
| `experiments/estimate.py` | What a configuration will cost, before it runs. |
| `tests/test_harness.py` | Offline. No AWS. Verifies the accounting arithmetic against AWS's own worked example. |

## Running

```bash
make test                     # offline, free
make estimate                 # upper-bound cost of a configuration
make e1 e2 e3                 # live, costs money
```

## Status

Harness complete and tested offline. No live results yet.
