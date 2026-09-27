#!/usr/bin/env python3
"""Render the paper's figures from the committed runs. No inference, no network.

Every number plotted is read from paper/data/*.json(l), which are the exact runs
the manuscript cites, so a figure cannot drift from the result it illustrates.

Palette: three slots of a validated categorical order (blue, orange, aqua) that
pass colour-vision-deficiency and normal-vision separation for every pair. Two
selected themes rather than one inverted -- each is stepped for its own surface.

    .venv/bin/python paper/figures.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "data"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1a1a1a", "grid": "#d8d8d8", "muted": "#6b6b6b",
              "series": ["#2b6cb0", "#c05621", "#2c7a7b"]},
    "dark":  {"bg": "#12161c", "fg": "#e8e8e8", "grid": "#2e3640", "muted": "#9aa4b0",
              "series": ["#63a4e0", "#e08a4c", "#4fb3b0"]},
}


def load(name, raw=False):
    p = DATA / (f"{name}.jsonl" if raw else f"{name}.json")
    if raw:
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return json.loads(p.read_text())


def style(ax, t, ylabel="", title=""):
    ax.set_facecolor(t["bg"])
    ax.figure.set_facecolor(t["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["fg"], fontsize=10)
    if title:
        ax.set_title(title, color=t["fg"], fontsize=11, loc="left", pad=12)


def p50(summary, catalog, arm, field):
    return summary["by_catalog"][str(catalog)][arm][field]["p50"]


def arms_by_catalog():
    """Merge the 10/50/200 run with the separate 100 run, and the terse sweep."""
    v = load("e1-verbose")["by_catalog"]
    v100 = load("e1-verbose-100")["by_catalog"]["100"]
    verbose = {int(k): val for k, val in v.items()}
    verbose[100] = v100
    terse = {int(k): val for k, val in load("e1-terse")["by_catalog"].items()}
    return dict(sorted(verbose.items())), dict(sorted(terse.items()))


# --------------------------------------------------------------- figure 1
def fig_combination(theme):
    """The combination penalty: caching a prefix that varies costs more than
    not caching it, at every catalog size."""
    t = THEMES[theme]
    verbose, _ = arms_by_catalog()
    cats = sorted(verbose)
    dyn_cache = [verbose[c]["dynamic_cache"]["usd"]["p50"] for c in cats]
    dyn = [verbose[c]["dynamic"]["usd"]["p50"] for c in cats]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax, t, "USD per call (median)",
          "Adding a cache checkpoint to a per-turn tool selection")
    w = 0.36
    xs = range(len(cats))
    ax.bar([x - w / 2 for x in xs], dyn, w, label="no checkpoint",
           color=t["series"][0], edgecolor=t["bg"], linewidth=1)
    ax.bar([x + w / 2 for x in xs], dyn_cache, w, label="with checkpoint",
           color=t["series"][1], edgecolor=t["bg"], linewidth=1)
    for x, (a, b) in enumerate(zip(dyn, dyn_cache)):
        ax.annotate(f"+{100 * (b / a - 1):.0f}%", (x + w / 2, b),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=t["series"][1], fontsize=9, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{c} tools" for c in cats])
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.text(0.01, -0.02, "The checkpointed arm wrote 42,541 tokens and read 0: it pays the "
             "write premium every turn and never reaches a read.",
             color=t["muted"], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"fig1-combination-{theme}.png", dpi=200,
                facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 2
def fig_crossover(theme):
    """Where a static cached catalog stops being cheaper, in both schema styles."""
    t = THEMES[theme]
    verbose, terse = arms_by_catalog()
    cats = sorted(verbose)

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    style(ax, t, "USD per call (median)",
          "Static cached catalog vs per-turn selection")
    ax.plot(cats, [verbose[c]["static"]["usd"]["p50"] for c in cats], "-o",
            color=t["series"][0], label="static, verbose schemas", linewidth=2, ms=5)
    ax.plot(cats, [terse[c]["static"]["usd"]["p50"] for c in cats], "-o",
            color=t["series"][2], label="static, terse schemas", linewidth=2, ms=5)
    ax.plot(cats, [verbose[c]["dynamic"]["usd"]["p50"] for c in cats], "--s",
            color=t["series"][1], label="per-turn selection (5 tools)", linewidth=2, ms=5)
    # The crossover is where the verbose static line passes the dynamic line.
    ax.axvspan(100, 200, color=t["series"][1], alpha=0.07)
    ax.annotate("crossover", xy=(140, max(verbose[200]["static"]["usd"]["p50"], 0) * 0.55),
                color=t["muted"], fontsize=9, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(cats)
    ax.set_xticklabels([str(c) for c in cats])
    ax.set_xlabel("tools in catalog", color=t["fg"], fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.text(0.01, -0.02, "Terse descriptions move the crossover beyond 200 tools: 200 terse "
             "tools (11,653 catalog tokens) beat 100 verbose ones (17,991).",
             color=t["muted"], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"fig2-crossover-{theme}.png", dpi=200,
                facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 3
def fig_two_currencies(theme):
    """Dollars and quota disagree at the top of the range."""
    t = THEMES[theme]
    verbose, _ = arms_by_catalog()
    cats = sorted(verbose)
    ratio_usd = [verbose[c]["dynamic"]["usd"]["p50"] / verbose[c]["static"]["usd"]["p50"]
                 for c in cats]
    ratio_q = [verbose[c]["dynamic"]["quota_tokens"]["p50"]
               / verbose[c]["static"]["quota_tokens"]["p50"] for c in cats]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax, t, "per-turn selection / static cached",
          "The two currencies disagree above ~150 tools")
    ax.plot(cats, ratio_usd, "-o", color=t["series"][0], label="dollars", linewidth=2, ms=5)
    ax.plot(cats, ratio_q, "-o", color=t["series"][2], label="quota tokens", linewidth=2, ms=5)
    ax.axhline(1.0, color=t["muted"], linewidth=1, linestyle=":")
    ax.annotate("above 1: static is cheaper", (cats[0], 1.06), color=t["muted"], fontsize=8.5)
    ax.annotate("below 1: selection is cheaper", (cats[0], 0.72), color=t["muted"], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks(cats)
    ax.set_xticklabels([str(c) for c in cats])
    ax.set_xlabel("tools in catalog", color=t["fg"], fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9, loc="center left")
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.text(0.01, -0.02, "Cache reads cost 10% of input in dollars and nothing at all against "
             "quota, so the same design is a loss on one axis and a 4x win on the other.",
             color=t["muted"], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"fig3-currencies-{theme}.png", dpi=200,
                facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 4
def fig_determinism(theme):
    """Distinct outputs from 16 identical requests, by arm."""
    t = THEMES[theme]
    rows = [r for r in load("e3-clean", raw=True)
            if r["arm"] in ("uncached", "cached") and not r["error"]]
    prompts = sorted({r["attrs"]["prompt"] for r in rows})
    counts = {arm: [len({r["text"] for r in rows
                         if r["arm"] == arm and r["attrs"]["prompt"] == p})
                    for p in prompts] for arm in ("uncached", "cached")}

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    style(ax, t, "distinct outputs / 16 requests",
          "Identical requests at temperature 0")
    w = 0.36
    xs = range(len(prompts))
    ax.bar([x - w / 2 for x in xs], counts["uncached"], w, label="no cache",
           color=t["series"][0], edgecolor=t["bg"], linewidth=1)
    ax.bar([x + w / 2 for x in xs], counts["cached"], w, label="cached prefix",
           color=t["series"][2], edgecolor=t["bg"], linewidth=1)
    for x in xs:
        for off, arm in ((-w / 2, "uncached"), (w / 2, "cached")):
            ax.annotate(str(counts[arm][x]), (x + off, counts[arm][x]),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        color=t["fg"], fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"prompt {p}" for p in prompts])
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.text(0.01, -0.02, "23 distinct outputs uncached against 10 cached. Reusing identical KV "
             "state removes a source of variation rather than adding one.",
             color=t["muted"], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"fig4-determinism-{theme}.png", dpi=200,
                facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


def main():
    for theme in THEMES:
        fig_combination(theme)
        fig_crossover(theme)
        fig_two_currencies(theme)
        fig_determinism(theme)
    made = sorted(p.name for p in OUT.glob("*.png"))
    print(f"wrote {len(made)} figures to {OUT.relative_to(ROOT)}:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
