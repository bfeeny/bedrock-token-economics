#!/usr/bin/env python3
"""E3, judged by Bedrock's managed evaluation service instead of by hand.

E3 asks whether a cached response says the same thing as an uncached one. The
first implementation answers that with our own Converse calls. This one hands
the same question to **Amazon Bedrock model evaluation** with a custom
LLM-as-a-judge metric, which is the managed way to do it.

Two reasons it is worth running both:

  * It is a fair test of the managed path. The hand-rolled judge and the
    service are asked the identical question over the identical pairs, so
    their verdicts can be compared rather than assumed equivalent.
  * `precomputedInferenceSource` lets the job score responses that already
    exist, so the generations from E3 are reused rather than paid for twice.

The dataset maps naturally: the uncached response is the reference, the cached
response is the candidate, and the metric asks whether the candidate says the
same thing.

Note on AgentCore: its Evaluate API takes OpenTelemetry session spans and is
built to judge agent trajectories, not to compare two strings. It has no
built-in evaluators enabled on this account. Bedrock evaluation is the right
managed surface for this question; AgentCore's is for a different one.

    python3 experiments/e3_managed_eval.py --submit
    python3 experiments/e3_managed_eval.py --status <jobArn>
"""
import argparse
import datetime
import glob
import itertools
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

STACK = "tokenecon-eval"
METRIC = {
    "name": "SemanticEquivalence",
    "instructions": (
        "You are comparing two responses to the same question.\n\n"
        "Question:\n{{prompt}}\n\n"
        "Response A (reference):\n{{ground_truth}}\n\n"
        "Response B (candidate):\n{{prediction}}\n\n"
        "Answer SAME if the two responses convey the same substantive content -- "
        "the same claims, facts and recommendations -- even if wording, ordering "
        "or formatting differ. Answer DIFFERENT if either states something "
        "substantive the other does not, or if they disagree.\n\n"
        "Reply with only SAME or DIFFERENT."
    ),
    "ratingScale": [
        {"definition": "The responses convey the same substantive content.",
         "value": {"stringValue": "SAME"}},
        {"definition": "The responses differ in substance.",
         "value": {"stringValue": "DIFFERENT"}},
    ],
}


def outputs(session, stack=STACK):
    d = session.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    return {o["OutputKey"]: o["OutputValue"] for o in d}


def build_dataset(pairs_per_prompt: int, seed: int) -> list[dict]:
    """Pair each uncached response with a cached one, from the latest E3 run."""
    files = sorted(glob.glob("results/raw/e3_cache_divergence-*.jsonl"))
    if not files:
        raise SystemExit("no E3 raw results found -- run experiments/e3_cache_divergence.py first")
    rows = [json.loads(l) for l in open(files[-1]) if l.strip()]
    # A generation can come back empty (stop reason, filter, or an empty
    # completion). The evaluation service rejects the whole dataset for one
    # blank referenceResponse, naming only the record number, so drop them here.
    rows = [r for r in rows if not r["error"] and r["arm"] in ("uncached", "cached")
            and (r.get("text") or "").strip()]
    if not rows:
        raise SystemExit(f"{files[-1]} has no usable generations")

    prompts = {}
    for r in rows:
        prompts.setdefault(r["attrs"]["prompt"], {}).setdefault(r["arm"], []).append(r["text"])

    rng = random.Random(seed)
    records = []
    for pid, arms in sorted(prompts.items()):
        if not (arms.get("uncached") and arms.get("cached")):
            continue
        combos = [(a, b) for a, b in itertools.product(arms["uncached"], arms["cached"])]
        rng.shuffle(combos)
        for a, b in [(x, y) for x, y in combos if x.strip() and y.strip()][:pairs_per_prompt]:
            records.append({
                "prompt": f"(prompt {pid}) Do these two responses say the same thing?",
                "referenceResponse": a,
                "category": f"prompt-{pid}",
                "modelResponses": [{"response": b, "modelIdentifier": "cached-arm"}],
            })
    print(f"  built {len(records)} pairs from {files[-1]}")
    return records


def submit(args):
    import boto3
    s = boto3.Session(profile_name=args.profile, region_name="us-east-1")
    out = outputs(s)
    bucket, role = out["BucketName"], out["RoleArn"]
    # jobName must match [a-z0-9](-*[a-z0-9]){0,62} -- no uppercase, no "T".
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    key = f"datasets/e3-pairs-{stamp}.jsonl"

    records = build_dataset(args.pairs, args.seed)
    body = "\n".join(json.dumps(r) for r in records).encode()
    s.client("s3").put_object(Bucket=bucket, Key=key, Body=body)
    print(f"  uploaded s3://{bucket}/{key}")

    r = s.client("bedrock").create_evaluation_job(
        jobName=f"e3-semantic-equivalence-{stamp}",
        jobDescription="Does a cached response say the same thing as an uncached one?",
        roleArn=role,
        applicationType="ModelEvaluation",
        inferenceConfig={"models": [
            # The responses already exist; the job only scores them.
            {"precomputedInferenceSource": {"inferenceSourceIdentifier": "cached-arm"}}
        ]},
        outputDataConfig={"s3Uri": f"s3://{bucket}/results/"},
        evaluationConfig={"automated": {
            "datasetMetricConfigs": [{
                # The service requires "General" for Model-as-a-Judge, and
                # "General" is NOT in the API model's own enum, which lists
                # Summarization | Classification | QuestionAndAnswer |
                # Generation | Custom. "Custom" and "Generation" are both
                # rejected. Passing a value botocore does not know about works
                # because it does not enforce enums client-side.
                "taskType": "General",
                "dataset": {"name": "e3-pairs",
                            "datasetLocation": {"s3Uri": f"s3://{bucket}/{key}"}},
                "metricNames": [METRIC["name"]],
            }],
            "customMetricConfig": {
                "customMetrics": [{"customMetricDefinition": METRIC}],
                "evaluatorModelConfig": {"bedrockEvaluatorModels": [
                    {"modelIdentifier": args.judge}]},
            },
        }},
    )
    print(f"\n  jobArn: {r['jobArn']}")
    print(f"  poll:   python3 experiments/e3_managed_eval.py --status {r['jobArn']}")
    return r["jobArn"]


def status(args):
    import boto3
    s = boto3.Session(profile_name=args.profile, region_name="us-east-1")
    j = s.client("bedrock").get_evaluation_job(jobIdentifier=args.status)
    print(f"  status : {j['status']}")
    if j.get("failureMessages"):
        for m in j["failureMessages"]:
            print(f"  failure: {m}")
    if j["status"] == "Completed":
        uri = j["outputDataConfig"]["s3Uri"]
        print(f"  results: {uri}")
        bucket = uri.split("/")[2]
        prefix = "/".join(uri.split("/")[3:])
        s3 = s.client("s3")
        keys = [o["Key"] for o in
                s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
                if o["Key"].endswith(".jsonl")]
        verdicts, explanations = {}, []
        for k in keys:
            for line in s3.get_object(Bucket=bucket, Key=k)["Body"].read().decode().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Results live under automatedEvaluationResult.scores[], not
                # under a top-level "results" key.
                for score in (rec.get("automatedEvaluationResult", {}) or {}).get("scores", []):
                    v = str(score.get("result", "")).upper()
                    verdicts[v] = verdicts.get(v, 0) + 1
                    for d in score.get("evaluatorDetails", []) or []:
                        if d.get("explanation"):
                            explanations.append(d["explanation"])
        if verdicts:
            total = sum(verdicts.values())
            print(f"  verdicts over {total} pairs:")
            for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
                print(f"    {v or '(blank)':10} {n:4}  {n/total:.1%}")
            # The managed evaluator returns prose reasoning per verdict. A
            # one-word Converse judge does not, and that is the clearest
            # argument for the managed path on an auditable question.
            if explanations:
                print(f"\n  sample reasoning ({len(explanations)} explanations returned):")
                print("    " + explanations[0][:280].replace("\n", " ") + "...")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", metavar="JOB_ARN")
    ap.add_argument("--pairs", type=int, default=20, help="pairs per prompt")
    ap.add_argument("--judge", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="personal")
    args = ap.parse_args()
    if args.status:
        return status(args)
    if args.submit:
        submit(args)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
