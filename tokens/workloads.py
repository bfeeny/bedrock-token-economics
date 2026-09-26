"""Prompt and tool-catalog builders for the experiments.

Kept separate from the experiments so that the thing being varied is obvious:
a tool catalog is parameterized by *count* and by *schema verbosity*, which the
literature routinely conflates, and any claim about "fewer tools" that does not
hold verbosity fixed is measuring two things at once.
"""
from __future__ import annotations

from .prices import model as price_of

# Rough, and deliberately conservative: used only to size a prefix so it clears
# a model's minimum cacheable length with margin, never to report a count.
CHARS_PER_TOKEN = 4.2

TOOL_DOMAINS = [
    ("crm", "customer record"), ("billing", "invoice"), ("inventory", "stock item"),
    ("hr", "employee record"), ("ticketing", "support ticket"), ("calendar", "event"),
    ("docs", "document"), ("email", "message"), ("analytics", "report"),
    ("deploy", "release"), ("monitoring", "alert"), ("storage", "file"),
]
VERBS = ["get", "list", "create", "update", "delete", "search", "archive", "export"]


def tool_spec(i: int, verbose: bool = True) -> dict:
    """One tool definition. `verbose` controls description length only, so tool
    count and schema size can be varied independently."""
    domain, noun = TOOL_DOMAINS[i % len(TOOL_DOMAINS)]
    verb = VERBS[(i // len(TOOL_DOMAINS)) % len(VERBS)]
    name = f"{domain}_{verb}_{i}"
    if verbose:
        desc = (f"{verb.capitalize()} a {noun} in the {domain} system. Use this tool when the "
                f"user asks to {verb} a {noun}, or refers to {domain} data by name, identifier "
                f"or date range. Returns the full {noun} record including metadata, audit "
                f"fields and related identifiers. Do not use for bulk operations.")
        params = {
            "identifier": {"type": "string",
                           "description": f"The unique identifier of the {noun}."},
            "include_metadata": {"type": "boolean",
                                 "description": "Whether to include audit and metadata fields."},
            "as_of": {"type": "string",
                      "description": "ISO-8601 timestamp to read the record as of."},
        }
    else:
        desc = f"{verb} a {domain} {noun}."
        params = {"identifier": {"type": "string", "description": "id"}}
    return {"toolSpec": {"name": name, "description": desc,
                         "inputSchema": {"json": {"type": "object", "properties": params,
                                                  "required": ["identifier"]}}}}


def tool_config(n: int, verbose: bool = True) -> dict:
    return {"tools": [tool_spec(i, verbose) for i in range(n)]}


def system_blocks(text: str, cache: bool = False, ttl: str = "5m") -> list:
    """A system prompt, optionally ending in a cache checkpoint.

    The checkpoint goes last because everything before it is what gets cached,
    and because Bedrock chains `tools` -> `system` -> `messages`: anything that
    changes earlier invalidates everything after it.
    """
    blocks: list = [{"text": text}]
    if cache:
        blocks.append({"cachePoint": {"type": "default", "ttl": ttl}})
    return blocks


def filler(paragraphs: int, seed_text: str = "") -> str:
    """Deterministic, non-repetitive filler to reach a target prefix length.

    Non-repetitive on purpose: a prefix of one sentence repeated compresses and
    tokenizes unlike real context, which would flatter any caching result.

    Deliberately neutral technical content. An earlier version described case
    review, analyst escalation and discrepancy handling, and asking the model to
    summarize risks in it produced `stop_reason: content_filtered` on a third of
    all generations -- empty strings that a judge then scored as identical to
    each other. Benign-looking text is not the same as text that survives a
    content filter, and a filtered response is an empty result, not a short one.
    """
    out = []
    for i in range(paragraphs):
        out.append(
            f"Section {i + 1}. {seed_text}Service tier {i % 7 + 1} maintains a connection pool "
            f"of {i * 4 + 12} workers and flushes its write buffer every {i % 9 + 2} seconds. "
            f"The scheduler assigns partition {i * 13 % 97} to the replica group holding lease "
            f"{i * 31 + 5}, and emits a checkpoint once {i * 17 + 40} records have been "
            f"acknowledged. Retries use exponential backoff starting at {i % 5 + 1} hundred "
            f"milliseconds with a ceiling of {i % 11 + 4} seconds, and the compaction job runs "
            f"when the segment count for shard {chr(97 + i % 26)}{i % 10} exceeds "
            f"{i % 6 + 3} times the configured floor.")
    return "\n\n".join(out)


def filler_for_model(model_id: str, margin: float = 1.25, minimum_paragraphs: int = 20) -> str:
    """Filler long enough that this model will actually cache the prefix.

    Observed live: an identical 3,935-token prefix caches on Sonnet 4.5
    (minimum 1,024) and is silently ignored by Haiku 4.5 (minimum 4,096) --
    same code, no error, full input billed. Sizing the prefix from the model's
    own minimum turns that from a result you have to notice into one you cannot
    produce by accident.
    """
    m = price_of(model_id)
    if not m.min_cache_tokens:
        return filler(minimum_paragraphs)
    target_chars = m.min_cache_tokens * margin * CHARS_PER_TOKEN
    paragraphs = minimum_paragraphs
    while len(filler(paragraphs)) < target_chars:
        paragraphs += 5
    return filler(paragraphs)
