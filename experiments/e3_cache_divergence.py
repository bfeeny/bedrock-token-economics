#!/usr/bin/env python3
"""E3 — Does caching change the answer, in any way that matters?

The first version of this experiment asked whether the cached and uncached
outputs differ, and could not answer, because **the uncached control is not
deterministic**. On Sonnet 4.5 at temperature 0, twelve identical requests
produced eight distinct outputs on one prompt. The published work this replicates
measures against a cache-off arm that is bit-identical across 800 episodes on
self-hosted stacks; a hosted API gives you no such baseline, so a diff cannot
attribute anything.

The answerable question is comparative. Two responses to the same prompt differ
somehow; the question is whether they differ *more* when one of them came from a
cache than when both came from the same arm. That makes the experiment a
two-sample comparison rather than a diff:

    within-uncached  uncached vs uncached  -- the model's own variability
    within-cached    cached vs cached      -- the same, inside the cached arm
    between          uncached vs cached    -- the test

The second control is what separates "caching shifted the distribution" from
"the cached arm is simply more variable". Only if each arm is internally
consistent *and* the arms disagree with each other has caching changed the
answer.

If between-arm disagreement is no higher than within-arm, caching has no
detectable effect on the answer, and the measurement says so with a number
rather than a shrug.

Equivalence is judged by a model, not by string equality, because string
equality would count "Three risks:" against "The three risks are:" as a
difference and swamp the effect being measured.

    python3 experiments/e3_cache_divergence.py --repeats 16 --pairs 24
"""
import argparse
import collections
import itertools
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tokens.meter import Meter  # noqa: E402
from tokens.stats import paired_bootstrap  # noqa: E402
from tokens.workloads import filler_for_model, system_blocks  # noqa: E402

# Open-ended, because a one-word answer cannot diverge: measuring divergence on
# a constrained task measures the task, not the cache.
# Neutral generative tasks over neutral text. The previous set asked for
# "risks implied by the procedure", which the content filter rejected outright.
PROMPTS = [
    "Summarize the configuration described above in exactly three sentences.",
    "Describe how the retry and backoff settings above relate to each other.",
    "Write a short paragraph explaining the compaction trigger above to a new engineer.",
]

JUDGE = (
    "Two responses to the same question follow. Answer SAME if they convey the "
    "same substantive content -- the same claims, facts and recommendations -- "
    "even if the wording, ordering or formatting differs. Answer DIFFERENT if "
    "either one states something substantive the other does not, or they "
    "disagree. Reply with one word: SAME or DIFFERENT.\n\n"
    "RESPONSE A:\n{a}\n\nRESPONSE B:\n{b}"
)


def judge_same(meter: Meter, model: str, a: str, b: str, seed: int, kind: str) -> bool | None:
    row = meter.call(arm=f"judge-{kind}", seed=seed, model_id=model,
                     messages=[{"role": "user",
                                "content": [{"text": JUDGE.format(a=a[:2500], b=b[:2500])}]}],
                     max_tokens=5, temperature=0.0, keep_text=True,
                     attrs={"kind": kind})
    if row.error:
        return None
    return row.text.strip().upper().startswith("SAME")


def run(args):
    meter = Meter("e3_cache_divergence", client=args.client, profile=args.profile)
    system_text = "You are a careful analyst.\n\n" + filler_for_model(args.model)

    # ---- generate
    for arm, cached in (("uncached", False), ("cached", True)):
        for i, prompt in enumerate(PROMPTS):
            for rep in range(args.repeats):
                meter.call(arm=arm, seed=rep, model_id=args.model,
                           messages=[{"role": "user", "content": [{"text": prompt}]}],
                           system=system_blocks(system_text, cache=cached),
                           max_tokens=args.max_tokens, temperature=0.0,
                           cache_expected=cached, attrs={"prompt": i, "rep": rep})

    # A filtered response is an empty string, and empty strings compare equal to
    # each other, so a filtered run reads as perfect determinism if unchecked.
    filtered = [r for r in meter.rows
                if r.arm in ("uncached", "cached") and r.stop_reason == "content_filtered"]
    if filtered:
        by_prompt = {}
        for r in filtered:
            by_prompt[r.attrs.get("prompt")] = by_prompt.get(r.attrs.get("prompt"), 0) + 1
        print(f"!! {len(filtered)} of "
              f"{len([r for r in meter.rows if r.arm in ('uncached','cached')])} generations were "
              f"content_filtered, by prompt: {by_prompt}")
        print("!! A filtered response is empty, and empty strings compare equal, so these")
        print("!! would read as perfect determinism. Change the prompts; do not report this.")
        return {"content_filtered": by_prompt}, meter

    problems = meter.assert_cache_behaved()
    if problems:
        print("!! " + "\n!! ".join(problems))
        print("!! Without a confirmed cache hit this experiment proves nothing.")
        return {"cache_assertions": problems}, meter

    # ---- compare
    rng = random.Random(args.seed)
    summary = {"model": args.model, "judge": args.judge, "repeats": args.repeats,
               "pairs_per_cell": args.pairs, "by_prompt": {}}
    for i in range(len(PROMPTS)):
        texts = {arm: [r.text for r in meter.rows
                       if r.arm == arm and r.attrs.get("prompt") == i and not r.error]
                 for arm in ("uncached", "cached")}
        cell = {"distinct": {a: len(set(t)) for a, t in texts.items()},
                "n": {a: len(t) for a, t in texts.items()}}

        cells = {
            "within_uncached": list(itertools.combinations(texts["uncached"], 2)),
            "within_cached": list(itertools.combinations(texts["cached"], 2)),
            "between": [(a, b) for a in texts["uncached"] for b in texts["cached"]],
        }
        for v in cells.values():
            rng.shuffle(v)

        verdicts = {}
        for kind, pairs in ((k, v[:args.pairs]) for k, v in cells.items()):
            v = [judge_same(meter, args.judge, a, b, j, kind)
                 for j, (a, b) in enumerate(pairs)]
            v = [x for x in v if x is not None]
            verdicts[kind] = v
            cell[f"{kind}_same_rate"] = round(sum(v) / len(v), 3) if v else None
            cell[f"{kind}_n"] = len(v)

        # Does adding a cache make two responses less alike than two responses
        # from the same arm already are?
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
        for base in ("within_uncached", "within_cached"):
            cell[f"between_minus_{base}"] = paired_bootstrap(
                [float(x) for x in verdicts[base]],
                [float(x) for x in verdicts["between"]], statistic=mean)
        summary["by_prompt"][i] = cell
        sig = [b for b in ("within_uncached", "within_cached")
               if cell[f"between_minus_{b}"].get("significant")]
        print(f"  prompt {i}: SAME-rate  within-uncached {cell['within_uncached_same_rate']}"
              f"  within-cached {cell['within_cached_same_rate']}"
              f"  between {cell['between_same_rate']}"
              f"   | differs from: {', '.join(sig) if sig else 'neither'}")

    judged = [r for r in meter.rows if r.arm.startswith("judge")]
    summary["judge_cost_usd"] = round(sum(r.usd or 0 for r in judged), 6)
    summary["judge_calls"] = len(judged)
    return summary, meter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--judge", default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    help="a cheaper model than the one under test")
    ap.add_argument("--repeats", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=24, help="judged pairs per cell")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
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
