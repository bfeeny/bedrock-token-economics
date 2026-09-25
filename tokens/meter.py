"""One Bedrock call, fully accounted.

Everything this harness claims rests on what `Meter.call` records, so it records
more than any single experiment needs:

  * the four token counts Bedrock reports separately -- input, output, cache
    read, cache write -- never a blended total, because the whole argument is
    that they are priced and quota'd differently;
  * both currencies, dollars and burndown-weighted quota tokens;
  * `stopReason`, because a `max_tokens` truncation bills the wasted output
    *and* the retry, and nobody has published what that costs;
  * whether a cache checkpoint actually took. A checkpoint below the model's
    minimum prefix is ignored and the request still succeeds, so an experiment
    that assumes caching happened will quietly measure nothing.

Rows go to JSONL. Analysis never re-runs inference.
"""
from __future__ import annotations

import json
import pathlib
import time
import uuid
from dataclasses import asdict, dataclass, field

from .prices import Model, dollars, model as price_of, quota_tokens


@dataclass
class Row:
    run_id: str
    experiment: str
    arm: str
    seed: int
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usd: float | None = None
    quota_tokens: float = 0.0
    latency_ms: float = 0.0
    stop_reason: str = ""
    truncated: bool = False
    cache_expected: bool = False
    cache_took: bool = False
    text: str = ""
    error: str = ""
    attrs: dict = field(default_factory=dict)


class Meter:
    def __init__(self, experiment: str, out_dir: str = "results/raw",
                 client=None, region: str = "us-east-1", profile: str | None = "personal"):
        self.experiment = experiment
        self.run_id = f"{experiment}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.path = pathlib.Path(out_dir) / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._region, self._profile = region, profile
        self.rows: list[Row] = []

    @property
    def client(self):
        if self._client is None:
            import boto3
            session = (boto3.Session(profile_name=self._profile, region_name=self._region)
                       if self._profile else boto3.Session(region_name=self._region))
            self._client = session.client("bedrock-runtime")
        return self._client

    def call(self, *, arm: str, seed: int, model_id: str, messages: list,
             system: list | None = None, tool_config: dict | None = None,
             max_tokens: int = 512, temperature: float = 0.0,
             effort: str | None = None, cache_expected: bool = False,
             keep_text: bool = True, attrs: dict | None = None) -> Row:
        m: Model = price_of(model_id)
        kwargs: dict = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            kwargs["system"] = system
        if tool_config:
            kwargs["toolConfig"] = tool_config
        if effort:
            kwargs["outputConfig"] = {"effort": effort}

        row = Row(run_id=self.run_id, experiment=self.experiment, arm=arm, seed=seed,
                  model_id=model_id, cache_expected=cache_expected, attrs=attrs or {})
        t0 = time.perf_counter()
        try:
            r = self.client.converse(**kwargs)
        except Exception as exc:  # noqa: BLE001 - a failed call is a data point
            row.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            row.error = f"{type(exc).__name__}: {exc}"[:300]
            self._write(row)
            return row

        row.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        u = r.get("usage", {})
        row.input_tokens = int(u.get("inputTokens", 0))
        row.output_tokens = int(u.get("outputTokens", 0))
        row.cache_read_tokens = int(u.get("cacheReadInputTokens", 0) or 0)
        row.cache_write_tokens = int(u.get("cacheWriteInputTokens", 0) or 0)
        row.stop_reason = r.get("stopReason", "")
        row.truncated = row.stop_reason == "max_tokens"
        row.cache_took = bool(row.cache_read_tokens or row.cache_write_tokens)
        row.usd = dollars(m, row.input_tokens, row.output_tokens,
                          row.cache_read_tokens, row.cache_write_tokens)
        row.quota_tokens = quota_tokens(m, row.input_tokens, row.output_tokens,
                                        row.cache_read_tokens, row.cache_write_tokens)
        if keep_text:
            parts = (r.get("output", {}).get("message", {}) or {}).get("content", [])
            row.text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        self._write(row)
        return row

    def _write(self, row: Row) -> None:
        self.rows.append(row)
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(row)) + "\n")

    # ------------------------------------------------------------------ checks
    def assert_cache_behaved(self) -> list[str]:
        """Cache checkpoints fail silently. Any arm that expected caching and
        never saw a read or a write measured something other than what it
        claims, so the run says so rather than reporting a null result."""
        bad = []
        for arm in sorted({r.arm for r in self.rows if r.cache_expected}):
            arm_rows = [r for r in self.rows if r.arm == arm and not r.error]
            if arm_rows and not any(r.cache_took for r in arm_rows):
                bad.append(f"{arm}: expected caching, saw no cache read or write in "
                           f"{len(arm_rows)} calls -- check the minimum prefix for this model")
        return bad
