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
    # Multipliers on the *input* rate, read off the published per-model table.
    # The 5-minute write is 1.25x and the 1-hour write is 2.0x on every
    # Anthropic model currently listed; the read discount is 0.10x except on
    # Opus 5.5, which is 0.05x. Both are overridable because neither is a rule.
    cache_write_mult: float = 1.25
    cache_write_mult_1h: float = 2.00
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
    # Anthropic rates: AWS Bedrock pricing page, us-east-1, read 2026-09-26.
    # Burndown and minimum cacheable prefix: AWS docs, read 2026-09-24.
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": Model(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        input_per_1k=0.001, output_per_1k=0.005,
        output_burndown=5.0, min_cache_tokens=4096,
        source="price: AWS pricing page us-east-1 2026-09-26; "
               "burndown+min-prefix: AWS docs 2026-09-24"),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": Model(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        input_per_1k=0.003, output_per_1k=0.015,
        output_burndown=5.0, min_cache_tokens=1024,
        source="price: AWS pricing page us-east-1 2026-09-26; "
               "burndown+min-prefix: AWS docs 2026-09-24"),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": Model(
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        input_per_1k=0.003, output_per_1k=0.015,
        # The pricing table lists no 1-hour cache write for Sonnet 4.
        cache_write_mult_1h=float("nan"),
        output_burndown=5.0, min_cache_tokens=1024,
        source="price: AWS pricing page us-east-1 2026-09-26 (no 1h cache write listed)"),
    "us.anthropic.claude-opus-4-5-20251101-v1:0": Model(
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        input_per_1k=0.005, output_per_1k=0.025,
        output_burndown=5.0, min_cache_tokens=4096,
        source="price: AWS pricing page us-east-1 2026-09-26; docs 2026-09-24"),
    "global.anthropic.claude-opus-5-5": Model(
        "global.anthropic.claude-opus-5-5",
        input_per_1k=0.004, output_per_1k=0.020,
        # Opus 5.5 reads at 95% off, not the 90% every other row shows.
        cache_read_mult=0.05,
        output_burndown=10.0, min_cache_tokens=512,
        source="price: AWS pricing page us-east-1 2026-09-26; docs 2026-09-24. "
               "NOT callable on this account (AccessDenied 2026-09-25)"),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": Model(
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        input_per_1k=0.006, output_per_1k=0.030,
        output_burndown=5.0, min_cache_tokens=1024,
        source="price+burndown+min-prefix: AWS pricing page & docs 2026-09-24"),
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
            cache_read: int = 0, cache_write: int = 0,
            ttl: str = "5m") -> float | None:
    """Dollars for one call. `ttl` selects the cache-write multiplier: a
    1-hour entry is written at 2.0x input against 1.25x for 5 minutes, so a
    long TTL needs proportionally more reads before it pays for itself."""
    if m.input_per_1k is None or m.output_per_1k is None:
        return None
    write_mult = m.cache_write_mult_1h if ttl == "1h" else m.cache_write_mult
    # Cache reads and writes are input tokens at a different rate. Bedrock
    # reports them separately from `inputTokens`, so they are added, not netted.
    return round(
        (input_tok * m.input_per_1k
         + cache_write * m.input_per_1k * write_mult
         + cache_read * m.input_per_1k * m.cache_read_mult
         + output_tok * m.output_per_1k) / 1000.0, 10)


def quota_tokens(m: Model, input_tok: int, output_tok: int,
                 cache_read: int = 0, cache_write: int = 0) -> float:
    """`input + cacheWrite + output x burndown`. Cache reads are free here --
    that asymmetry is why caching is a throughput lever, not only a cost one."""
    del cache_read
    return input_tok + cache_write + output_tok * m.output_burndown


def cache_breakeven_reads(m: Model, ttl: str = "5m") -> float:
    """How many reads a cached prefix needs before it has paid for its write.

    A write costs (mult - 1) extra input rates; each read saves (1 - read_mult).
    Below this many reads the cache is a loss, which is why cache *hit rate*,
    not cache existence, is the variable that matters.
    """
    write_mult = m.cache_write_mult_1h if ttl == "1h" else m.cache_write_mult
    return (write_mult - 1.0) / (1.0 - m.cache_read_mult)


# ---------------------------------------------------------------- AgentCore
# Verified from the AgentCore pricing page, read 2026-09-26. These matter to E1
# because the dynamic-tool arm does not only pay a cache penalty: picking the
# tools costs a Search API call per turn, and holding the catalog costs a
# monthly indexing fee. Comparing the mechanism alone would flatter it.
GATEWAY_INVOCATION_USD = 0.005 / 1000      # ListTools, InvokeTool, Ping
GATEWAY_SEARCH_USD = 0.025 / 1000          # semantic tool search, per call
GATEWAY_INDEX_USD_PER_TOOL_MONTH = 0.02 / 100


def gateway_fees(searches: int = 0, invocations: int = 0,
                 tools_indexed: int = 0) -> float:
    """Per-run Gateway fees. Indexing is monthly, so it is reported separately
    by callers rather than amortized into a per-turn number here."""
    return (searches * GATEWAY_SEARCH_USD
            + invocations * GATEWAY_INVOCATION_USD
            + tools_indexed * GATEWAY_INDEX_USD_PER_TOOL_MONTH)
