"""Prices and quota burndown, with the provenance of every number attached.

Two currencies come out of a Bedrock call and they are not proportional:

  dollars  input + output at per-model rates, with cache writes at a premium
           and cache reads at a steep discount.
  quota    input + cacheWrite + (output x burndown). Cache *reads* do not
           count at all. Burndown is 5x to 15x depending on the model, so an
           output token can cost fifteen input tokens of throughput capacity.

A technique that improves one can worsen the other, which is why every result
in this harness reports both. Anything unpriced yields a token ledger with the
dollar column omitted -- never a guessed rate.
"""
from __future__ import annotations

from dataclasses import dataclass

PRICED_ON = "2026-09-24"


@dataclass(frozen=True)
class Model:
    model_id: str
    # USD per 1,000 tokens. None where the rate has not been verified.
    input_per_1k: float | None = None
    output_per_1k: float | None = None
    # Multipliers on the *input* rate. 1.25x write / 0.10x read is what AWS
    # documents for GPT-5.6 and what the Claude 3.5 Sonnet v2 price table works
    # out to exactly; treat as the default and override where verified.
    cache_write_mult: float = 1.25
    cache_read_mult: float = 0.10
    # Output tokens consumed per output token billed, against TPM/TPD quota.
    # Verified: docs.aws.amazon.com/bedrock/latest/userguide/quotas-token-burndown.html
    output_burndown: float = 1.0
    # Smallest prefix that can be cached. Below this the checkpoint is ignored
    # and the request still succeeds -- silently. Assert, never assume.
    min_cache_tokens: int | None = None
    source: str = ""


# Burndown rates are verified from the AWS token-burndown page (read 2026-09-24).
# Dollar rates are verified only where `source` says so; the rest are None on
# purpose, because a plausible-looking wrong price is worse than no price.
MODELS: dict[str, Model] = {
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": Model(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        output_burndown=5.0, min_cache_tokens=4096,
        source="burndown+min-prefix: AWS docs 2026-09-24; price UNVERIFIED"),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": Model(
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        input_per_1k=0.006, output_per_1k=0.030,
        output_burndown=5.0, min_cache_tokens=1024,
        source="price+burndown+min-prefix: AWS pricing page & docs 2026-09-24"),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": Model(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        output_burndown=5.0, min_cache_tokens=1024,
        source="burndown+min-prefix: AWS docs 2026-09-24; price UNVERIFIED "
               "(absent from the Price List API 2026-09-25)"),
    "mistral.ministral-3-3b-instruct": Model(
        "mistral.ministral-3-3b-instruct",
        input_per_1k=0.0001, output_per_1k=0.0001,
        output_burndown=1.0,
        source="price: repo price table 2026-09-20; burndown 1:1 (non-Anthropic)"),
    "mistral.mistral-large-3-675b-instruct": Model(
        "mistral.mistral-large-3-675b-instruct",
        input_per_1k=0.0005, output_per_1k=0.0015,
        output_burndown=1.0,
        source="price: repo price table 2026-09-20; burndown 1:1 (non-Anthropic)"),
}


def model(model_id: str) -> Model:
    """Unknown models still meter; they simply have no dollar column."""
    return MODELS.get(model_id, Model(model_id, source="UNKNOWN MODEL - tokens only"))


def dollars(m: Model, input_tok: int, output_tok: int,
            cache_read: int = 0, cache_write: int = 0) -> float | None:
    if m.input_per_1k is None or m.output_per_1k is None:
        return None
    # Cache reads and writes are input tokens at a different rate. Bedrock
    # reports them separately from `inputTokens`, so they are added, not netted.
    return round(
        (input_tok * m.input_per_1k
         + cache_write * m.input_per_1k * m.cache_write_mult
         + cache_read * m.input_per_1k * m.cache_read_mult
         + output_tok * m.output_per_1k) / 1000.0, 10)


def quota_tokens(m: Model, input_tok: int, output_tok: int,
                 cache_read: int = 0, cache_write: int = 0) -> float:
    """`input + cacheWrite + output x burndown`. Cache reads are free here --
    that asymmetry is why caching is a throughput lever, not only a cost one."""
    del cache_read
    return input_tok + cache_write + output_tok * m.output_burndown
