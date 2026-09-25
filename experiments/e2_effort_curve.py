#!/usr/bin/env python3
"""E2 — What does `effort` actually cost and buy?

`outputConfig.effort` is GA on Bedrock, documented, and tells Claude "how
liberally it should spend tokens." Anthropic publishes no effort-versus-accuracy
curve and the docs tell you to sweep it on your own evals. Two independent
research passes found zero published measurements. So this measures one.

It matters more on Bedrock than the raw token counts suggest, because output
tokens burn quota at 5x to 15x depending on the model while input burns at 1x.
A technique that trades input for output is a bad trade here even when the
total token count improves, which is why every arm reports both currencies.

Accuracy uses exact-match on a checkable task set, so "cheaper" can never be
reported without saying whether it was also worse. Cost-matched comparison is
the point: accuracy at equal spend, not spend at equal accuracy.

    python3 experiments/e2_effort_curve.py --efforts low medium high --seeds 3
"""
import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tokens.meter import Meter  # noqa: E402
from tokens.stats import describe, paired_bootstrap  # noqa: E402

# Small, checkable, and deliberately mixed in difficulty: the accuracy cost of
# lower effort grows with task hardness, so a single-difficulty set would give a
# flat curve and the wrong conclusion.
TASKS = [
    ("A store sells pens at $3 each. If Maria buys 7 pens and pays with a $50 note, "
     "how much change does she get? Reply with the number only.", "29"),
    ("A train leaves at 14:45 and the journey takes 2 hours 40 minutes. "
     "What time does it arrive, in 24-hour HH:MM? Reply with the time only.", "17:25"),
    ("What is 17 * 23? Reply with the number only.", "391"),
    ("A tank holds 240 litres and is 3/8 full. How many litres are in it? "
     "Reply with the number only.", "90"),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines "
     "need to make 100 widgets? Reply with the number only.", "5"),
    ("A book has 310 pages. Ana reads 28 pages a day for 9 days. How many pages "
     "remain? Reply with the number only.", "58"),
    ("What is the sum of the first 20 positive integers? Reply with the number only.", "210"),
    ("A shirt costs $40 after a 20% discount. What was the original price in dollars? "
     "Reply with the number only.", "50"),
]


def graded(text: str, expected: str) -> bool:
    """Accept the answer in any reasonable wrapper, but require the value.

    Deliberately permissive about formatting and strict about the number: a
    grader that punishes formatting would confound effort with verbosity, which
    is exactly the variable under test.
    """
    if not text:
        return False
    cleaned = text.strip().replace(",", "").replace("$", "")
    if ":" in expected:
        return expected in cleaned
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return bool(nums) and (expected in nums)


def run(args):
    meter = Meter("e2_effort_curve", client=args.client, profile=args.profile)
    for effort in args.efforts:
        for seed in range(args.seeds):
            for i, (prompt, expected) in enumerate(TASKS):
                row = meter.call(
                    arm=effort, seed=seed, model_id=args.model,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    max_tokens=args.max_tokens, effort=effort,
                    attrs={"task": i, "expected": expected})
                row.attrs["correct"] = graded(row.text, expected)

    summary = {"model": args.model, "seeds": args.seeds, "n_tasks": len(TASKS), "arms": {}}
    baseline = args.efforts[-1]
    base_rows = [r for r in meter.rows if r.arm == baseline and not r.error]
    for effort in args.efforts:
        rows = [r for r in meter.rows if r.arm == effort and not r.error]
        correct = [r for r in rows if r.attrs.get("correct")]
        summary["arms"][effort] = {
            "accuracy": round(len(correct) / len(rows), 4) if rows else None,
            "output_tokens": describe([float(r.output_tokens) for r in rows]),
            "quota_tokens": describe([r.quota_tokens for r in rows]),
            "usd": describe([r.usd for r in rows if r.usd is not None]),
            "latency_ms": describe([r.latency_ms for r in rows]),
            "truncated": sum(r.truncated for r in rows),
            # Against the highest-effort arm, so the trade is explicit.
            "quota_vs_baseline": paired_bootstrap(
                [r.quota_tokens for r in base_rows], [r.quota_tokens for r in rows]),
        }
        a = summary["arms"][effort]
        print(f"  {effort:8} acc {a['accuracy']}  output p50 {a['output_tokens'].get('p50')}"
              f"  quota p50 {a['quota_tokens'].get('p50')}  truncated {a['truncated']}")

    trunc = sum(r.truncated for r in meter.rows)
    if trunc:
        print(f"\n!! {trunc} responses hit max_tokens. A truncation bills the wasted output "
              f"AND the retry, at the output burndown rate -- raise --max-tokens or report it.")
    return summary, meter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    ap.add_argument("--efforts", nargs="+", default=["low", "medium", "high"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
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
