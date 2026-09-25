#!/usr/bin/env python3
"""What will a run cost, before it runs.

The account this runs in has a small monthly budget, and E1 in particular
multiplies out fast: catalogs x arms x seeds x turns. This prices a
configuration from token estimates so a bad parameter choice is caught here
rather than on the bill.

Estimates are deliberately pessimistic on input size (it assumes the cache never
hits) so the number quoted is an upper bound.

    python3 experiments/estimate.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tokens.prices import dollars, model as price_of  # noqa: E402
from tokens.workloads import filler, tool_config  # noqa: E402

CHARS_PER_TOKEN = 3.6   # conservative for English prose with punctuation


def toks(s: str) -> int:
    return int(len(s) / CHARS_PER_TOKEN)


def estimate(model_id: str, calls: int, in_tok: int, out_tok: int, label: str) -> float:
    m = price_of(model_id)
    usd = dollars(m, in_tok, out_tok)
    if usd is None:
        print(f"  {label:34} {calls:5} calls  ~{in_tok:6} in /{out_tok:5} out   "
              f"cost UNKNOWN (model unpriced)")
        return 0.0
    total = usd * calls
    print(f"  {label:34} {calls:5} calls  ~{in_tok:6} in /{out_tok:5} out   ${total:8.4f}")
    return total


def main() -> int:
    haiku = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    sonnet = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    system_tok = toks(filler(60))
    print(f"system prefix used by E1/E3: ~{system_tok} tokens "
          f"(Haiku 4.5 needs >=4096 to cache at all)\n")
    if system_tok < 4096:
        print("  !! below Haiku 4.5's minimum cacheable prefix -- raise --filler-paragraphs\n")

    total = 0.0
    print("E1 tool catalog x cache  (3 catalogs, 2 arms, 3 seeds, 6 turns):")
    for n in (10, 50, 200):
        tool_tok = toks(str(tool_config(n)))
        total += estimate(sonnet, 2 * 3 * 6, system_tok + tool_tok, 128, f"catalog {n}")
    print("\nE2 effort curve  (3 efforts, 3 seeds, 8 tasks):")
    total += estimate(sonnet, 3 * 3 * 8, 60, 700, "reasoning tasks")
    print("\nE3 cache divergence  (2 arms, 3 prompts, 15 repeats):")
    total += estimate(sonnet, 2 * 3 * 15, system_tok, 300, "open-ended generation")

    print(f"\n  upper bound, priced on Sonnet 3.5 v2 and assuming no cache hits: ${total:.2f}")
    print(f"  {haiku} is unpriced in the table, so a Haiku run cannot be quoted -- "
          "verify its rate before using it for a costed claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
