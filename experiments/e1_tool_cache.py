#!/usr/bin/env python3
"""E1 — Does retrieving tools cost more than caching them?

AWS recommends two things that fight each other and never says so.

Well-Architected advises narrowing the tool set per request, and AgentCore
Gateway ships semantic tool search to do it. Separately, Bedrock prompt caching
processes checkpoints `tools` -> `system` -> `messages`, and *changing an
earlier section invalidates the cache for the later ones*. So a tool set that
varies per request re-bills the entire prefix at cache-write rates on every
turn, while a static catalog is written once and read thereafter -- and cache
reads do not count against the token quota at all.

Whichever wins depends on the catalog size, the turns per session and the
schema verbosity. This finds the crossover.

  static        full catalog, cache checkpoint after tools+system. Big prefix,
                written once, read on every later turn.
  dynamic-cache a plausible subset per turn, *with* a checkpoint -- what a team
                gets by following both AWS recommendations at once. The prefix
                changes every turn, so it writes every turn and reads never,
                paying the 1.25x write premium for nothing.
  dynamic       the same varying subset with no checkpoint at all. The control
                that separates "the tool set varies" from "a checkpoint was
                present", which the first version of this experiment conflated.

    python3 experiments/e1_tool_cache.py --tools 10 50 200 --turns 6 --seeds 5
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tokens.meter import Meter  # noqa: E402
from tokens.prices import gateway_fees  # noqa: E402
from tokens.stats import describe, paired_bootstrap  # noqa: E402
from tokens.workloads import filler_for_model, system_blocks, tool_config  # noqa: E402

TURN_PROMPTS = [
    "Look up the customer record for identifier C-4471 and summarize it in one line.",
    "What was the invoice total for identifier INV-9920?",
    "Check stock for item SKU-3312 and say whether it needs reordering.",
    "Find the support ticket T-8801 and give its current status.",
    "Show the calendar event EV-5150 and who is attending.",
    "Retrieve document D-7729 and give its title.",
]


def run(args) -> dict:
    meter = Meter("e1_tool_cache", client=args.client, profile=args.profile)
    # A system prompt long enough to clear the model's minimum cacheable prefix.
    # Haiku 4.5 needs 4,096 tokens; below that the checkpoint is silently ignored.
    base_system = ("You are an operations assistant with access to internal tools. "
                   "Answer concisely.\n\n" + filler_for_model(args.model))

    def system_for(seed: int, n_tools: int) -> str:
        """A marker unique to (run, catalog, seed), so nothing shares a cache
        entry with anything it should not.

        Three contaminations this prevents, all observed:
          * seeds warming each other -- seed 0 writes, seeds 1-2 read, and every
            arm appears to amortize;
          * catalogs warming each other within a run;
          * **runs warming each other**. Cache entries outlive the process for
            the TTL, so a benchmark re-run inside five minutes reads what the
            previous run wrote. That showed up here as an arm reporting zero
            writes and forty thousand reads, which is impossible in isolation.
        """
        return f"Session {meter.run_id}/{n_tools}/{seed:04d}.\n\n" + base_system

    for n in args.tools:
        if n - args.subset < 2:
            raise SystemExit(
                f"catalog {n} with subset {args.subset} leaves no room to vary: "
                f"the dynamic arm would send the same tools every turn and "
                f"silently behave like the static arm. Use --subset < {n - 1}.")

    summary = {"model": args.model, "turns": args.turns, "seeds": args.seeds,
               "verbose_schemas": args.verbose_schemas,
               "filler_paragraphs": args.filler_paragraphs, "by_catalog": {}}

    for n_tools in args.tools:
        for arm in ("static", "dynamic-cache", "dynamic"):
            for seed in range(args.seeds):
                for turn in range(args.turns):
                    prompt = TURN_PROMPTS[turn % len(TURN_PROMPTS)]
                    if arm == "static":
                        cfg = tool_config(n_tools, verbose=args.verbose_schemas)
                    else:
                        # A different plausible subset per turn: this is what
                        # semantic tool selection produces, and it is precisely
                        # what breaks the prefix.
                        span = max(1, n_tools - args.subset)
                        lo = (turn * 3 + seed * 7) % span
                        cfg = {"tools": tool_config(n_tools, args.verbose_schemas)["tools"]
                               [lo:lo + args.subset]}
                    wants_cache = arm in ("static", "dynamic-cache")
                    meter.call(
                        arm=f"{arm}-{n_tools}", seed=seed, model_id=args.model,
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                        system=system_blocks(system_for(seed, n_tools), cache=wants_cache),
                        tool_config=cfg, max_tokens=args.max_tokens,
                        cache_expected=wants_cache,
                        keep_text=False,
                        attrs={"turn": turn, "n_tools": n_tools,
                               "tools_sent": len(cfg["tools"])})

        rows = [r for r in meter.rows if r.attrs.get("n_tools") == n_tools and not r.error]
        # An arm that reads without ever writing read someone else's entry.
        for label, arm_rows in (("static", [r for r in rows if r.arm == f"static-{n_tools}"]),
                                ("dynamic-cache",
                                 [r for r in rows if r.arm == f"dynamic-cache-{n_tools}"])):
            reads = sum(r.cache_read_tokens for r in arm_rows)
            writes = sum(r.cache_write_tokens for r in arm_rows)
            if reads and not writes:
                print(f"  !! {label}-{n_tools} read {reads} tokens having written none -- "
                      f"cache contamination from another run; results are not isolated")
        # The dynamic arm pays one Gateway search per turn to choose its tools.
        # Reported alongside the model cost, never silently folded into it.
        n_dynamic_turns = len([r for r in rows if r.arm.startswith("dynamic")])
        search_fee = gateway_fees(searches=n_dynamic_turns)
        index_fee_month = gateway_fees(tools_indexed=n_tools)
        st = [r for r in rows if r.arm == f"static-{n_tools}"]
        dc = [r for r in rows if r.arm == f"dynamic-cache-{n_tools}"]
        dy = [r for r in rows if r.arm == f"dynamic-{n_tools}"]
        summary["by_catalog"][n_tools] = {
            "static": _arm_summary(st), "dynamic_cache": _arm_summary(dc),
            "dynamic": _arm_summary(dy),
            "usd_delta_dynamic_minus_static": paired_bootstrap(
                [r.usd or 0 for r in st], [r.usd or 0 for r in dy]),
            "quota_delta_dynamic_minus_static": paired_bootstrap(
                [r.quota_tokens for r in st], [r.quota_tokens for r in dy]),
            "gateway_search_fee_usd_this_run": round(search_fee, 6),
            "gateway_index_fee_usd_per_month": round(index_fee_month, 4),
        }
        for label, arm_rows in (("static", st), ("dynamic+cache", dc), ("dynamic", dy)):
            print(f"  catalog {n_tools:>3} {label:14} p50 ${_p50(arm_rows, 'usd'):.6f}"
                  f"  quota {_p50(arm_rows, 'quota_tokens'):7.0f}"
                  f"  writes {sum(r.cache_write_tokens for r in arm_rows):7}"
                  f"  reads {sum(r.cache_read_tokens for r in arm_rows):8}")

    problems = meter.assert_cache_behaved()
    summary["cache_assertions"] = problems or "ok"
    if problems:
        print("\n!! " + "\n!! ".join(problems))
    return summary, meter


def _arm_summary(rows) -> dict:
    return {
        "usd": describe([r.usd for r in rows if r.usd is not None]),
        "quota_tokens": describe([r.quota_tokens for r in rows]),
        "input_tokens": describe([float(r.input_tokens) for r in rows]),
        "cache_read": describe([float(r.cache_read_tokens) for r in rows]),
        "cache_write": describe([float(r.cache_write_tokens) for r in rows]),
        "latency_ms": describe([r.latency_ms for r in rows]),
    }


def _p50(rows, field) -> float:
    vals = [getattr(r, field) or 0 for r in rows]
    return round(describe(vals).get("p50", float("nan")), 6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    ap.add_argument("--tools", type=int, nargs="+", default=[10, 50, 200])
    ap.add_argument("--subset", type=int, default=5, help="tools the dynamic arm sends per turn")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--filler-paragraphs", type=int, default=60,
                    help="pads the system prompt past the model's minimum cacheable prefix")
    # Verbosity is the variable the literature conflates with tool count, so it
    # has to be independently settable. (It previously could not be turned off.)
    ap.add_argument("--terse-schemas", dest="verbose_schemas", action="store_false",
                    default=True, help="one-line tool descriptions instead of full ones")
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
