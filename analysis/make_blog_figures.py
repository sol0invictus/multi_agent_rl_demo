"""Figures for the multi-agent RL blog post, built directly from the runs' logs.

    ../rl_harness/.venv-lfm/bin/python analysis/make_blog_figures.py --out blog/multi-agent-rl

Reads runs/org_small/ and runs/org_small_lead/ ({metrics,eval,eval_rollouts,train_rollouts}.jsonl)
and rebuilds the test set from org_tasks.make_org_test to recover each test task's difficulty and mess.
Writes PNGs to --out and analysis/results_org_small.json with every number quoted in the post.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from org_tasks import make_org_test  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run-all", default="org_small")
p.add_argument("--run-lead", default="org_small_lead")
p.add_argument("--out", default=os.path.join(ROOT, "blog", "multi-agent-rl"))
p.add_argument("--n-test", type=int, default=10)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

# Reference palette from sunnyguha.com / rl_harness blog
COLOR = {"all": "#2a78d6", "lead": "#eb6834", "solo": "#1baf7a"}
INK, INK2, MUTED, GRID, BASE = "#1a1a1a", "#52514e", "#8a8984", "#e6e5e1", "#b9b8b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "#fdfdfd",
    "axes.facecolor": "#fdfdfd", "savefig.facecolor": "#fdfdfd",
})


def jl(run, name):
    with open(os.path.join(ROOT, "runs", run, name)) as f:
        return [json.loads(line) for line in f if line.strip()]


met_all = jl(args.run_all, "metrics.jsonl")
met_lead = jl(args.run_lead, "metrics.jsonl")
eval_all = {e["step"]: e for e in jl(args.run_all, "eval.jsonl")}
eval_lead = {e["step"]: e for e in jl(args.run_lead, "eval.jsonl")}
er_all = {e["step"]: e["results"] for e in jl(args.run_all, "eval_rollouts.jsonl")}
er_lead = {e["step"]: e["results"] for e in jl(args.run_lead, "eval_rollouts.jsonl")}

STEPS_ALL = sorted(eval_all)
STEPS_LEAD = sorted(eval_lead)
S0, SF = STEPS_ALL[0], STEPS_ALL[-1]
TASKS = {t.task_id: t for t in make_org_test(args.n_test)}
results = {"steps_all": STEPS_ALL, "steps_lead": STEPS_LEAD, "final_step": SF}


def save(fig, name):
    fig.savefig(os.path.join(args.out, name))
    plt.close(fig)
    print("wrote", name)


def pct_axis(ax, lo=0.0, hi=1.0):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))


def sign_test(w, l):
    n = w + l
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0


# ------------------------------------------------------------- fig 1: held-out accuracy over training
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5), gridspec_kw={"width_ratios": [1.1, 1.1, 1.35]})

ax = axes[0]
ys_all_org = [eval_all[s]["org"]["acc"] for s in STEPS_ALL]
ys_lead_org = [eval_lead[s]["org"]["acc"] for s in STEPS_LEAD]
ax.plot(STEPS_ALL, ys_all_org, color=COLOR["all"], lw=2.2, marker="o", ms=6, mec="#fdfdfd", mew=1.5,
        label="train all roles (RLCR)")
ax.plot(STEPS_LEAD, ys_lead_org, color=COLOR["lead"], lw=2.0, ls="--", marker="o", ms=6, mec="#fdfdfd", mew=1.5,
        label="train lead only")
ax.annotate(f"{ys_all_org[-1]:.1%}", (SF, ys_all_org[-1]), xytext=(31.2, ys_all_org[-1]),
            va="center", fontsize=9.5, fontweight="bold", color=COLOR["all"])
ax.annotate(f"{ys_lead_org[-1]:.1%}", (SF, ys_lead_org[-1]), xytext=(31.2, ys_lead_org[-1]),
            va="center", fontsize=9.5, fontweight="bold", color=COLOR["lead"])
ax.set_title("Multi-agent org (n=40)")
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("training step")
ax.set_xlim(-2, 37)
pct_axis(ax, 0.05, 0.72)
ax.set_ylabel("held-out accuracy")
ax.legend(loc="upper left", fontsize=8.8)

ax = axes[1]
ys_all_solo = [eval_all[s]["solo"]["acc"] for s in STEPS_ALL]
ys_lead_solo = [eval_lead[s]["solo"]["acc"] for s in STEPS_LEAD]
ax.plot(STEPS_ALL, ys_all_solo, color=COLOR["all"], lw=2.2, marker="o", ms=6, mec="#fdfdfd", mew=1.5,
        label="train all roles -> solo")
ax.plot(STEPS_LEAD, ys_lead_solo, color=COLOR["lead"], lw=2.0, ls="--", marker="o", ms=6, mec="#fdfdfd", mew=1.5,
        label="train lead only -> solo")
ax.annotate(f"{ys_all_solo[-1]:.1%}", (SF, ys_all_solo[-1]), xytext=(31.2, ys_all_solo[-1]),
            va="center", fontsize=9.5, fontweight="bold", color=COLOR["all"])
ax.annotate(f"{ys_lead_solo[-1]:.1%}", (SF, ys_lead_solo[-1]), xytext=(31.2, ys_lead_solo[-1]),
            va="center", fontsize=9.5, fontweight="bold", color=COLOR["lead"])
ax.set_title("Zero-shot solo transfer (n=40)")
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("training step (trained on org only)")
ax.set_xlim(-2, 37)
pct_axis(ax, 0.05, 0.72)
ax.legend(loc="lower right", fontsize=8.8)

ax = axes[2]
cats = ["seen", "tmpl", "domain", "wide\n(K=4)", "org\nall", "solo\nall", "both\n(n=80)"]
v_base = [
    eval_all[S0]["org"]["acc_seen"], eval_all[S0]["org"]["acc_tmpl"],
    eval_all[S0]["org"]["acc_domain"], eval_all[S0]["org"]["acc_wide"],
    eval_all[S0]["org"]["acc"], eval_all[S0]["solo"]["acc"],
    0.5 * (eval_all[S0]["org"]["acc"] + eval_all[S0]["solo"]["acc"]),
]
v_lead = [
    eval_lead[SF]["org"]["acc_seen"], eval_lead[SF]["org"]["acc_tmpl"],
    eval_lead[SF]["org"]["acc_domain"], eval_lead[SF]["org"]["acc_wide"],
    eval_lead[SF]["org"]["acc"], eval_lead[SF]["solo"]["acc"],
    0.5 * (eval_lead[SF]["org"]["acc"] + eval_lead[SF]["solo"]["acc"]),
]
v_all = [
    eval_all[SF]["org"]["acc_seen"], eval_all[SF]["org"]["acc_tmpl"],
    eval_all[SF]["org"]["acc_domain"], eval_all[SF]["org"]["acc_wide"],
    eval_all[SF]["org"]["acc"], eval_all[SF]["solo"]["acc"],
    0.5 * (eval_all[SF]["org"]["acc"] + eval_all[SF]["solo"]["acc"]),
]
x = np.arange(len(cats))
w = 0.26
ax.bar(x - w, v_base, width=w, color=BASE, label=f"base (step {S0})")
ax.bar(x, v_lead, width=w, color=COLOR["lead"], label=f"lead only (step {SF})")
ax.bar(x + w, v_all, width=w, color=COLOR["all"], label=f"all roles (step {SF})")
ax.axvline(3.5, color=MUTED, ls=":", lw=1.0)
ax.set_title(f"Step {SF} by split & mode")
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=8.8)
pct_axis(ax, 0.0, 0.72)
ax.legend(loc="upper left", fontsize=8.3)

fig.subplots_adjust(wspace=0.26)
save(fig, "fig_eval_curves.png")
results["eval_all"] = eval_all
results["eval_lead"] = eval_lead

# ------------------------------------------------------------- fig 2: coordination & worker behavior
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.3), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})

ax = axes[0]
rows_plot = [
    ("base (step 0)", eval_all[0]["org"]["statuses"]),
    ("all roles (step 10)", eval_all[10]["org"]["statuses"]),
    ("all roles (step 20)", eval_all[20]["org"]["statuses"]),
    ("all roles (step 30)", eval_all[30]["org"]["statuses"]),
    ("lead only (step 30)", eval_lead[30]["org"]["statuses"]),
]
OUT = [
    ("submitted", "called submit()", COLOR["all"]),
    ("final_text", "replied in text", "#8fb9e8"),
    ("max_turns", "hit turn limit", BASE),
]
for i, (label, st) in enumerate(rows_plot):
    tot, left = sum(st.values()), 0.0
    for k, lab, col in OUT:
        w = st.get(k, 0) / tot
        ax.barh(i, w, left=left, color=col, edgecolor="#fdfdfd", lw=2, height=0.62, label=lab if i == 0 else None)
        if w >= 0.07:
            ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=8.8,
                    color="#fdfdfd" if k == "submitted" else INK)
        left += w
ax.set_yticks(range(len(rows_plot)))
ax.set_yticklabels([r[0] for r in rows_plot], fontsize=9.2)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(False)
ax.set_title("How org episodes end")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8.5)

ax = axes[1]
lat_all = [eval_all[s]["org"]["latency"] for s in STEPS_ALL]
lat_lead = [eval_lead[s]["org"]["latency"] for s in STEPS_LEAD]
ax.plot(STEPS_ALL, lat_all, color=COLOR["all"], lw=2.2, marker="o", ms=6, mec="#fdfdfd", mew=1.5,
        label="all roles: latency")
ax.plot(STEPS_LEAD, lat_lead, color=COLOR["lead"], lw=2.0, ls="--", marker="o", ms=6, mec="#fdfdfd", mew=1.5,
        label="lead only: latency")
ax.annotate(f"{lat_all[-1]:.1f}", (SF, lat_all[-1]), xytext=(31.2, lat_all[-1]), va="center",
            fontsize=9, fontweight="bold", color=COLOR["all"])
ax.annotate(f"{lat_lead[-1]:.1f}", (SF, lat_lead[-1]), xytext=(31.2, lat_lead[-1]), va="center",
            fontsize=9, fontweight="bold", color=COLOR["lead"])
ax.set_title("Critical-path latency (calls)")
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("training step")
ax.set_xlim(-2, 36)
ax.set_ylim(11, 20.5)
ax.legend(loc="lower left", fontsize=8.5)

ax = axes[2]
exh_all = [eval_all[s]["org"]["worker_exhausted"] for s in STEPS_ALL]
exh_lead = [eval_lead[s]["org"]["worker_exhausted"] for s in STEPS_LEAD]
ax.plot(STEPS_ALL, exh_all, color=COLOR["all"], lw=2.2, marker="o", ms=6, mec="#fdfdfd", mew=1.5,
        label="all roles (-55%)")
ax.plot(STEPS_LEAD, exh_lead, color=COLOR["lead"], lw=2.0, ls="--", marker="o", ms=6, mec="#fdfdfd", mew=1.5,
        label="lead only (+5%)")
ax.annotate(f"{exh_all[-1]:.2f}", (SF, exh_all[-1]), xytext=(31.2, exh_all[-1]), va="center",
            fontsize=9, fontweight="bold", color=COLOR["all"])
ax.annotate(f"{exh_lead[-1]:.2f}", (SF, exh_lead[-1]), xytext=(31.2, exh_lead[-1]), va="center",
            fontsize=9, fontweight="bold", color=COLOR["lead"])
ax.set_title("Exhausted workers per task")
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("training step")
ax.set_xlim(-2, 36)
ax.set_ylim(0.3, 1.35)
ax.legend(loc="lower left", fontsize=8.5)

fig.subplots_adjust(wspace=0.28)
save(fig, "fig_coordination.png")


# ------------------------------------------------------------- fig 3: dumbbell breakdown by task property
def acc_by(rollouts, key_fn):
    d = defaultdict(list)
    for r in rollouts:
        t = TASKS[r["task_id"]]
        k = key_fn(r, t)
        if k is not None:
            d[k].append(r["correct"])
    return {k: (sum(v) / len(v), len(v)) for k, v in d.items()}


groups = [
    ("By site count K (org)", lambda r, t: f"K={t.n_sites}" if r["mode"] == "org" else None, ["K=2", "K=3", "K=4"]),
    ("By difficulty (org + solo)", lambda r, t: t.difficulty, ["easy", "medium", "hard"]),
    ("By data mess (org + solo)",
     lambda r, t: "clean" if not t.mess else ("1 mess" if len(t.mess) == 1 else "2 messes"),
     ["clean", "1 mess", "2 messes"]),
]

fig, axes = plt.subplots(1, 3, figsize=(13.6, 2.9))
results["breakdown"] = {}
for ax, (title, key_fn, cats) in zip(axes, groups):
    a0 = acc_by(er_all[S0], key_fn)
    a_lead = acc_by(er_lead[SF], key_fn)
    a_all = acc_by(er_all[SF], key_fn)
    ys = list(range(len(cats)))[::-1]
    for y, c in zip(ys, cats):
        (v0, n), (vl, _), (va, _) = a0[c], a_lead[c], a_all[c]
        lo, hi = min(v0, vl, va), max(v0, vl, va)
        ax.plot([lo, hi], [y, y], color=BASE, lw=2, zorder=1)
        ax.scatter([v0], [y], s=60, color=BASE, zorder=2, edgecolor="#fdfdfd", lw=1.5)
        ax.scatter([vl], [y], s=60, color=COLOR["lead"], zorder=3, edgecolor="#fdfdfd", lw=1.5)
        ax.scatter([va], [y], s=65, color=COLOR["all"], zorder=4, edgecolor="#fdfdfd", lw=1.5)
        ax.annotate(f"{va:.0%}", (max(v0, vl, va), y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold", color=COLOR["all"])
        results["breakdown"][f"{title}/{c}"] = {"base": v0, "lead": vl, "all": va, "n": n}
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c}\n(n={a0[c][1]})" for c in cats], fontsize=9.2)
    ax.set_title(title)
    ax.set_xlim(-0.05, 0.92)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)

axes[0].scatter([], [], s=60, color=BASE, label=f"base (step {S0})")
axes[0].scatter([], [], s=60, color=COLOR["lead"], label=f"lead only (step {SF})")
axes[0].scatter([], [], s=65, color=COLOR["all"], label=f"all roles (step {SF})")
axes[0].legend(loc="upper left", fontsize=9, bbox_to_anchor=(0, -0.15), ncol=3)
fig.subplots_adjust(wspace=0.42)
save(fig, "fig_breakdown.png")


# ------------------------------------------------------------- fig 4: training curves + stability
def rolling(xs, k=5):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]


st_all = [m["step"] for m in met_all]
st_lead = [m["step"] for m in met_lead]
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.2))

ax = axes[0]
ys_a = [m["acc"] for m in met_all]
ys_l = [m["acc"] for m in met_lead]
ax.plot(st_all, ys_a, color=COLOR["all"], lw=0.8, alpha=0.28)
ax.plot(st_lead, ys_l, color=COLOR["lead"], lw=0.8, alpha=0.28)
ax.plot(st_all, rolling(ys_a, 5), color=COLOR["all"], lw=2.2, label="all roles (RLCR)")
ax.plot(st_lead, rolling(ys_l, 5), color=COLOR["lead"], lw=2.0, ls="--", label="lead only")
ax.axvline(29, color=MUTED, lw=1, ls=":")
ax.annotate("step 29: 12 errors\n(lead only)", (29, 0.56), xytext=(-82, 0), textcoords="offset points",
            fontsize=8, color=COLOR["lead"], va="top")
ax.set_title("Batch accuracy (5-step mean)")
ax.set_xlabel("training step")
pct_axis(ax, 0.0, 0.62)
ax.legend(loc="upper left", fontsize=8.8)

ax = axes[1]
ax.plot(st_all, [1e3 * m.get("mismatch_kl", 0) for m in met_all], color=COLOR["all"], lw=1.6, label="all roles")
ax.plot(st_lead, [1e3 * m.get("mismatch_kl", 0) for m in met_lead], color=COLOR["lead"], lw=1.4, ls="--",
        label="lead only")
ax.set_title("vLLM vs trainer KL per token (x1e-3)")
ax.set_xlabel("training step")
ax.set_ylim(-0.4, 1.0)
ax.legend(loc="upper right", fontsize=8.8)

ax = axes[2]
ax.plot([1, 6, 11, 16, 21, 26],
        [np.mean([m["flat_groups"] / 4 for m in met_all[i * 5:(i + 1) * 5]]) for i in range(6)],
        color=COLOR["all"], lw=2.0, marker="o", ms=5.5, label="all roles")
ax.plot([1, 6, 11, 16, 21, 26],
        [np.mean([m["flat_groups"] / 4 for m in met_lead[i * 5:(i + 1) * 5]]) for i in range(6)],
        color=COLOR["lead"], lw=2.0, marker="o", ms=5.5, label="lead only")
ax.set_title("Groups with no reward spread (5-step)")
ax.set_xlabel("training step window start")
pct_axis(ax, 0.0, 0.80)
ax.legend(loc="upper right", fontsize=8.8)

fig.subplots_adjust(wspace=0.26)
save(fig, "fig_training.png")

# ------------------------------------------------------------- paired significance numbers
def paired_stats(r_a, r_b):
    ka = {(r["task_id"], r["mode"]): r["correct"] for r in r_a}
    kb = {(r["task_id"], r["mode"]): r["correct"] for r in r_b}
    out = {}
    for m in ("org", "solo", "all"):
        ks = [k for k in ka if m == "all" or k[1] == m]
        w = sum(kb[k] and not ka[k] for k in ks)
        l = sum(ka[k] and not kb[k] for k in ks)
        out[m] = {"fixed": w, "broke": l, "p": sign_test(w, l), "n": len(ks)}
    return out


results["paired"] = {
    "all_0_to_30": paired_stats(er_all[S0], er_all[SF]),
    "lead_0_to_30": paired_stats(er_lead[S0], er_lead[SF]),
    "lead_30_to_all_30": paired_stats(er_lead[SF], er_all[SF]),
}
out_json = os.path.join(ROOT, "analysis", f"results_{args.run_all}.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"wrote analysis/results_{args.run_all}.json")
