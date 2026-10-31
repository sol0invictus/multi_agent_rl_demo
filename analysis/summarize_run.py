"""Markdown tables for a run: evals (org vs solo per bucket plus coordination stats) and the training curve.

    python analysis/summarize_run.py org_all [--every 5] [--compare org_lead]

--compare adds a paired test (exact McNemar) on the last eval of two runs, matching rollouts by task and mode.
Both runs must use the same eval tasks (same --n-test / --eval-buckets).
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from org_tasks import TEST_BUCKETS  # noqa: E402

ROOT = os.path.join(HERE, "runs")


def load(run, name):
    path = os.path.join(ROOT, run, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def eval_table(evals):
    modes = [m for m in ("org", "solo") if any(m in e for e in evals)]
    head = ["step"] + [f"{m} {b}" for m in modes for b in ("all", *TEST_BUCKETS)] + \
        ["org cov", "org full cov", "org acc│full cov", "org fanout", "org latency"]
    rows = []
    for e in evals:
        r = [str(e["step"])]
        for m in modes:
            d = e.get(m, {})
            r += [fmt(d.get("acc"))] + [fmt(d.get("acc_" + b)) for b in TEST_BUCKETS]
        o = e.get("org", {})
        r += [fmt(o.get(k), 2) for k in ("coverage", "full_coverage", "acc_given_full_coverage", "max_fanout")]
        r += [fmt(o.get("latency"), 1)]
        rows.append(r)
    return _md(head, rows)


def train_table(metrics, every):
    keys = ["reward", "acc", "latency", "coverage", "max_fanout", "asks", "flat_groups", "errors",
            "mismatch_kl", "grad_norm", "t_rollout", "t_train"]
    keys = [k for k in keys if any(k in m for m in metrics)]
    rows, chunk = [], []
    for m in metrics:
        chunk.append(m)
        if len(chunk) == every or m is metrics[-1]:
            row = [f"{chunk[0]['step']}-{chunk[-1]['step']}"]
            for k in keys:
                vals = [c[k] for c in chunk if k in c]
                row.append(fmt(sum(vals) / len(vals), 3 if k in ("reward", "acc", "mismatch_kl") else 2))
            rows.append(row)
            chunk = []
    return _md(["steps"] + keys, rows)


def mcnemar(a, b):
    """Exact two-sided McNemar p-value from the discordant counts."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(run_a, run_b):
    last = lambda run: load(run, "eval_rollouts.jsonl")[-1]
    ea, eb = last(run_a), last(run_b)
    key = lambda r: (r["task_id"], r["mode"])
    ra, rb = {key(r): r["correct"] for r in ea["results"]}, {key(r): r["correct"] for r in eb["results"]}
    out = []
    for mode in sorted({k[1] for k in ra}):
        ks = [k for k in ra if k[1] == mode and k in rb]
        fixed = sum(not ra[k] and rb[k] for k in ks)
        broke = sum(ra[k] and not rb[k] for k in ks)
        out.append([mode, str(len(ks)), fmt(sum(ra[k] for k in ks) / max(1, len(ks))),
                    fmt(sum(rb[k] for k in ks) / max(1, len(ks))), str(fixed), str(broke), f"{mcnemar(fixed, broke):.2g}"])
    return (f"{run_a} (step {ea['step']}) vs {run_b} (step {eb['step']})\n\n"
            + _md(["mode", "n", run_a, run_b, f"only {run_b} right", f"only {run_a} right", "p"], out))


def _md(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--every", type=int, default=5, help="training steps per row")
    p.add_argument("--compare", default="", help="another run to test against on the last eval")
    a = p.parse_args()
    evals, metrics = load(a.run, "eval.jsonl"), load(a.run, "metrics.jsonl")
    if not evals and not metrics:
        raise SystemExit(f"nothing logged in {os.path.join(ROOT, a.run)}")
    print(f"## {a.run}\n\n### Eval\n\n{eval_table(evals)}\n\n### Training\n\n{train_table(metrics, a.every)}")
    if a.compare:
        print("\n### Paired comparison\n\n" + compare(a.run, a.compare))
