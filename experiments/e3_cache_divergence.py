#!/usr/bin/env python3
"""E3 — Does a cache hit change the answer?

Prefix caching reuses KV state, and reusing it changes the order of
floating-point accumulation. Floating-point addition is not associative, so the
reused and recomputed paths can differ in the last bits, and when a difference
crosses a sampling boundary the sampled token changes. Research on open-weight
models under vLLM/llama.cpp measured 36% of agentic trajectories diverging at
FP16 and 75% at 4-bit.

Vendors state that cache hits do not change output quality. Both can be true:
the conditional distribution is unchanged while the sampled realization moves.

Nobody has published this for a hosted API. This settles it for Bedrock with our
own data: the same prompt at temperature 0, with and without a cache checkpoint,
repeated, and the outputs diffed.

    python3 experiments/e3_cache_divergence.py --repeats 20
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tokens.meter import Meter  # noqa: E402
from tokens.workloads import filler, system_blocks  # noqa: E402

# Open-ended generation, because a one-word answer cannot diverge: any
# divergence measurement on a constrained task measures the task, not the cache.
PROMPTS = [
    "Summarize the operating procedure above in exactly three sentences.",
    "List the three most important risks implied by the procedure above.",
    "Write a short paragraph explaining the escalation rule above to a new analyst.",
]


def run(args):
    meter = Meter("e3_cache_divergence", client=args.client, profile=args.profile)
    system_text = "You are a careful analyst.\n\n" + filler(args.filler_paragraphs)

    for arm, cached in (("uncached", False), ("cached", True)):
        for i, prompt in enumerate(PROMPTS):
            for rep in range(args.repeats):
                meter.call(arm=arm, seed=rep, model_id=args.model,
                           messages=[{"role": "user", "content": [{"text": prompt}]}],
                           system=system_blocks(system_text, cache=cached),
                           max_tokens=args.max_tokens, temperature=0.0,
                           cache_expected=cached, attrs={"prompt": i, "rep": rep})

    summary = {"model": args.model, "repeats": args.repeats, "by_prompt": {}}
    for i in range(len(PROMPTS)):
        out = {}
        for arm in ("uncached", "cached"):
            texts = [r.text for r in meter.rows
                     if r.arm == arm and r.attrs.get("prompt") == i and not r.error]
            counts = collections.Counter(texts)
            out[arm] = {
                "n": len(texts),
                "distinct_outputs": len(counts),
                # Self-consistency: identical request, identical settings.
                "modal_share": round(counts.most_common(1)[0][1] / len(texts), 3) if texts else None,
            }
        # Does the cached arm ever produce something the uncached arm never did?
        un = {r.text for r in meter.rows
              if r.arm == "uncached" and r.attrs.get("prompt") == i and not r.error}
        ca = {r.text for r in meter.rows
              if r.arm == "cached" and r.attrs.get("prompt") == i and not r.error}
        out["cached_only_outputs"] = len(ca - un)
        out["shared_outputs"] = len(ca & un)
        summary["by_prompt"][i] = out
        print(f"  prompt {i}: uncached {out['uncached']['distinct_outputs']} distinct, "
              f"cached {out['cached']['distinct_outputs']} distinct, "
              f"{out['cached_only_outputs']} cached-only")

    problems = meter.assert_cache_behaved()
    summary["cache_assertions"] = problems or "ok"
    if problems:
        print("\n!! " + "\n!! ".join(problems)
              + "\n!! Without a confirmed cache hit this experiment proves nothing.")
    return summary, meter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--filler-paragraphs", type=int, default=60)
    ap.add_argument("--profile", default="personal")
    ap.set_defaults(client=None)
    args = ap.parse_args()
    summary, meter = run(args)
    out = pathlib.Path("results") / f"{meter.run_id}.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    print(f"\nwrote {out} and {meter.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
