#!/usr/bin/env python3
"""Offline tests. No AWS calls: the Bedrock client is faked.

The harness makes claims about money and quota, so the arithmetic is tested
directly rather than eyeballed on a live run where a wrong multiplier would be
invisible.

    python3 tests/test_harness.py
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tokens import prices  # noqa: E402
from tokens.meter import Meter  # noqa: E402
from tokens.stats import describe, paired_bootstrap  # noqa: E402
from tokens.workloads import filler, system_blocks, tool_config  # noqa: E402

failures = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


class FakeBedrock:
    """Returns the usage block we ask it to, so accounting can be tested exactly."""
    def __init__(self, usage=None, text="4", stop="end_turn"):
        self.usage = usage or {"inputTokens": 100, "outputTokens": 10}
        self.text, self.stop, self.calls = text, stop, []

    def converse(self, **kw):
        self.calls.append(kw)
        return {"usage": self.usage, "stopReason": self.stop,
                "output": {"message": {"content": [{"text": self.text}]}}}


tmp = tempfile.mkdtemp()

# ---------------------------------------------------------------- accounting
haiku = prices.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
check("burndown is carried per model, not assumed 1:1", haiku.output_burndown == 5.0)
check("quota counts output at the burndown rate",
      prices.quota_tokens(haiku, 1000, 100) == 1000 + 100 * 5)
check("cache reads are free against quota; writes are not",
      prices.quota_tokens(haiku, 0, 0, cache_read=10_000, cache_write=0) == 0
      and prices.quota_tokens(haiku, 0, 0, cache_read=0, cache_write=10_000) == 10_000)

sonnet = prices.model("anthropic.claude-3-5-sonnet-20241022-v2:0")
check("AWS's own worked example reproduces: 1,000 in + 100 out = 1,500 quota",
      prices.quota_tokens(sonnet, 1000, 100) == 1500)
check("...while billing 1,100 tokens' worth of dollars",
      abs(prices.dollars(sonnet, 1000, 100)
          - (1000 * 0.006 + 100 * 0.030) / 1000) < 1e-12)
check("a cache write costs 1.25x input and a read 0.10x",
      abs(prices.dollars(sonnet, 0, 0, cache_write=1000) - 1.25 * 0.006) < 1e-12
      and abs(prices.dollars(sonnet, 0, 0, cache_read=1000) - 0.10 * 0.006) < 1e-12)
check("an unpriced model still meters tokens but reports no dollars",
      prices.dollars(prices.model("nonesuch"), 100, 10) is None
      and prices.quota_tokens(prices.model("nonesuch"), 100, 10) == 110)

# A cached turn must be cheaper than an uncached one on both currencies,
# otherwise the premise of E1 is wrong.
uncached_usd = prices.dollars(sonnet, 10_000, 50)
cached_usd = prices.dollars(sonnet, 100, 50, cache_read=9_900)
check("a cache read beats paying full input price", cached_usd < uncached_usd)
check("and the quota saving is larger than the dollar saving",
      (prices.quota_tokens(sonnet, 100, 50, cache_read=9_900)
       / prices.quota_tokens(sonnet, 10_000, 50)) < (cached_usd / uncached_usd))

# ---------------------------------------------------------------- the meter
fake = FakeBedrock(usage={"inputTokens": 50, "outputTokens": 20,
                          "cacheReadInputTokens": 4000, "cacheWriteInputTokens": 0})
m = Meter("t", out_dir=tmp, client=fake)
row = m.call(arm="a", seed=0, model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
             messages=[{"role": "user", "content": [{"text": "hi"}]}], cache_expected=True)
check("the four token counts are recorded separately, never blended",
      (row.input_tokens, row.output_tokens, row.cache_read_tokens, row.cache_write_tokens)
      == (50, 20, 4000, 0))
check("a confirmed cache hit is recorded as such", row.cache_took is True)
check("cache assertions pass when caching demonstrably happened", m.assert_cache_behaved() == [])

silent = FakeBedrock(usage={"inputTokens": 50, "outputTokens": 20})
m2 = Meter("t2", out_dir=tmp, client=silent)
m2.call(arm="static", seed=0, model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}], cache_expected=True)
check("a checkpoint that silently did not take is caught, not reported as a null result",
      len(m2.assert_cache_behaved()) == 1)

trunc = FakeBedrock(stop="max_tokens")
m3 = Meter("t3", out_dir=tmp, client=trunc)
r3 = m3.call(arm="a", seed=0, model_id="x",
             messages=[{"role": "user", "content": [{"text": "hi"}]}])
check("truncation is flagged, because it bills the waste and the retry", r3.truncated)

m4 = Meter("t4", out_dir=tmp, client=FakeBedrock())
m4.call(arm="a", seed=0, model_id="x", messages=[{"role": "user", "content": [{"text": "hi"}]}],
        effort="low", max_tokens=99, temperature=0.0)
sent = m4._client.calls[-1]
check("effort is sent as outputConfig.effort", sent["outputConfig"] == {"effort": "low"})
check("temperature 0 is passed through, so divergence is the cache's doing",
      sent["inferenceConfig"]["temperature"] == 0.0)

boom = FakeBedrock()
boom.converse = lambda **kw: (_ for _ in ()).throw(RuntimeError("throttled"))
m5 = Meter("t5", out_dir=tmp, client=boom)
r5 = m5.call(arm="a", seed=0, model_id="x", messages=[])
check("a failed call is recorded as a row, not lost", "RuntimeError" in r5.error)

rows = [line for line in (pathlib.Path(tmp) / f"{m.run_id}.jsonl").read_text().splitlines() if line]
check("every call is appended to JSONL so analysis never re-runs inference", len(rows) == 1)

# ---------------------------------------------------------------- statistics
check("describe reports the spread, not just the middle",
      describe([1, 1, 1, 100])["max_over_min"] == 100.0)
b = paired_bootstrap([1, 1, 1, 1, 1] * 6, [2, 2, 2, 2, 2] * 6, iters=500)
check("a real difference is called significant", b["delta"] == 1 and b["significant"])
b2 = paired_bootstrap([1, 2, 3, 4, 5] * 6, [5, 4, 3, 2, 1] * 6, iters=500)
check("no difference is not called significant", not b2["significant"])

# ---------------------------------------------------------------- workloads
cfg = tool_config(12, verbose=True)
lean = tool_config(12, verbose=False)
check("tool count and schema verbosity vary independently",
      len(cfg["tools"]) == len(lean["tools"]) == 12
      and len(str(cfg)) > 2 * len(str(lean)))
check("tool names are unique across a large catalog",
      len({t["toolSpec"]["name"] for t in tool_config(200)["tools"]}) == 200)
blocks = system_blocks("hello", cache=True, ttl="1h")
check("the cache checkpoint goes last, after the text it caches",
      blocks[0] == {"text": "hello"} and blocks[1]["cachePoint"]["ttl"] == "1h")
check("no checkpoint is added when caching is off", len(system_blocks("hello")) == 1)
f1, f2 = filler(3), filler(3)
check("filler is deterministic", f1 == f2)
check("filler is not repetitive, which would flatter a caching result",
      len(set(f1.split("\n\n"))) == 3)

print(f"\n{'FAILED' if failures else 'all passed'}: {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
