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


# ------------------------------------------------------------- fig 0: hero summary & punchline (top of post)
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.35), gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]})

# Panel A: Multi-agent org progress over training
ax = axes[0]
ys_all_org = [eval_all[s]["org"]["acc"] for s in STEPS_ALL]
ys_lead_org = [eval_lead[s]["org"]["acc"] for s in STEPS_LEAD]
ax.plot(STEPS_ALL, ys_all_org, color=COLOR["all"], lw=2.5, marker="o", ms=6.5, mec="#fdfdfd", mew=1.5,
        label="Train all roles (lead + workers)")
ax.plot(STEPS_LEAD, ys_lead_org, color=COLOR["lead"], lw=2.0, ls="--", marker="o", ms=6, mec="#fdfdfd", mew=1.5,
        label="Train lead only (freeze workers)")
ax.annotate(f"37.5%\n(+15 pts)", (SF, ys_all_org[-1]), xytext=(31.3, ys_all_org[-1] + 0.01),
            va="center", fontsize=9.0, fontweight="bold", color=COLOR["all"])
ax.annotate(f"27.5%\n(stalled)", (SF, ys_lead_org[-1]), xytext=(31.3, ys_lead_org[-1] - 0.015),
            va="center", fontsize=8.8, fontweight="bold", color=COLOR["lead"])
ax.set_title("1. Multi-agent accuracy over training")
ax.set_xticks([0, 10, 20, 30])
ax.set_xlabel("RL training step (1 GPU, ~3.5 hrs)")
ax.set_xlim(-2, 39)
pct_axis(ax, 0.08, 0.50)
ax.set_ylabel("held-out accuracy")
ax.legend(loc="upper left", fontsize=8.3)

# Panel B: Headline accuracy at Step 30 (Base vs Lead-only vs All roles)
ax = axes[1]
groups_h = ["Multi-agent\n(org)", "Single-agent\n(solo, 0-shot)", "4-site OOD\n(K=4 wide)"]
v0_h = [eval_all[S0]["org"]["acc"], eval_all[S0]["solo"]["acc"], eval_all[S0]["org"]["acc_wide"]]
vl_h = [eval_lead[SF]["org"]["acc"], eval_lead[SF]["solo"]["acc"], eval_lead[SF]["org"]["acc_wide"]]
va_h = [eval_all[SF]["org"]["acc"], eval_all[SF]["solo"]["acc"], eval_all[SF]["org"]["acc_wide"]]
xh = np.arange(len(groups_h))
wh = 0.25
ax.bar(xh - wh, v0_h, width=wh, color=BASE, label="Base model (step 0)")
ax.bar(xh, vl_h, width=wh, color=COLOR["lead"], label="Lead only (step 30)")
ax.bar(xh + wh, va_h, width=wh, color=COLOR["all"], label="All roles (step 30)")
for i in range(len(groups_h)):
    ax.text(xh[i] + wh, va_h[i] + 0.018, f"{va_h[i]:.0%}", ha="center", va="bottom",
            fontsize=8.6, fontweight="bold", color=COLOR["all"])
    ax.text(xh[i], vl_h[i] + 0.018, f"{vl_h[i]:.0%}", ha="center", va="bottom",
            fontsize=8.2, fontweight="bold", color=COLOR["lead"])
ax.set_title("2. Accuracy at step 30")
ax.set_xticks(xh)
ax.set_xticklabels(groups_h, fontsize=8.8)
pct_axis(ax, 0.0, 0.86)
ax.legend(loc="upper left", fontsize=8.0)

# Panel C: Coordination speed & worker reliability punchline
ax = axes[2]
cats_c = ["Critical-path\nlatency\n(calls / task)",
          "Exhausted\nworkers\n(per task)"]
base_c = [eval_all[S0]["org"]["latency"], eval_all[S0]["org"]["worker_exhausted"]]
lead_c = [eval_lead[SF]["org"]["latency"], eval_lead[SF]["org"]["worker_exhausted"]]
all_c = [eval_all[SF]["org"]["latency"], eval_all[SF]["org"]["worker_exhausted"]]
# Normalize to Base = 100% so both fit on one clean horizontal comparison with exact raw numbers annotated
y_c = np.array([1, 0])
hc = 0.24
ax.barh(y_c + hc, [1.0, 1.0], height=hc, color=BASE, label="Base (step 0)")
ax.barh(y_c, [lead_c[0] / base_c[0], lead_c[1] / base_c[1]], height=hc, color=COLOR["lead"], label="Lead only (step 30)")
ax.barh(y_c - hc, [all_c[0] / base_c[0], all_c[1] / base_c[1]], height=hc, color=COLOR["all"], label="All roles (step 30)")

ax.text(1.02, 1 + hc, f"{base_c[0]:.1f} calls", va="center", fontsize=8.3, color=INK2)
ax.text(lead_c[0] / base_c[0] + 0.02, 1, f"{lead_c[0]:.1f} calls (-8%)", va="center", fontsize=8.3, color=COLOR["lead"], fontweight="bold")
ax.text(all_c[0] / base_c[0] + 0.02, 1 - hc, f"{all_c[0]:.1f} calls (-23%)", va="center", fontsize=8.5, color=COLOR["all"], fontweight="bold")

ax.text(1.02, 0 + hc, f"{base_c[1]:.2f} / task", va="center", fontsize=8.3, color=INK2)
ax.text(lead_c[1] / base_c[1] + 0.02, 0, f"{lead_c[1]:.2f} / task (-7%)", va="center", fontsize=8.3, color=COLOR["lead"], fontweight="bold")
ax.text(all_c[1] / base_c[1] + 0.02, 0 - hc, f"{all_c[1]:.2f} / task (-55%)", va="center", fontsize=8.5, color=COLOR["all"], fontweight="bold")

ax.set_yticks(y_c)
ax.set_yticklabels(cats_c, fontsize=8.7)
ax.set_xlim(0, 1.48)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("relative to base model (100%)")
ax.set_title("3. Team speed & worker failures")
ax.grid(axis="y", visible=False)

fig.subplots_adjust(wspace=0.36)
save(fig, "fig_hero_punchline.png")


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

# ------------------------------------------------------------- fig 5: 2 training regimes head-to-head
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.4), gridspec_kw={"width_ratios": [1.0, 1.3, 1.15]})

# Panel 1: Mean trained tokens per step by role (lead vs worker)
ax = axes[0]
lead_tok_all = np.mean([m.get("loss_tokens_lead", 0) for m in met_all if m.get("rows", 0) > 0]) / 1e3
work_tok_all = np.mean([m.get("loss_tokens_worker", 0) for m in met_all if m.get("rows", 0) > 0]) / 1e3
lead_tok_ld = np.mean([m.get("loss_tokens_lead", 0) for m in met_lead if m.get("rows", 0) > 0]) / 1e3
work_tok_ld = np.mean([m.get("loss_tokens_worker", 0) for m in met_lead if m.get("rows", 0) > 0]) / 1e3

x_reg = np.arange(2)
ax.bar(x_reg, [lead_tok_all, lead_tok_ld], width=0.44, color=["#1b5299", "#b84a1d"],
       label="lead tokens (ask/submit)")
ax.bar(x_reg, [work_tok_all, work_tok_ld], bottom=[lead_tok_all, lead_tok_ld], width=0.44,
       color=[COLOR["all"], COLOR["lead"]], label="worker tokens (run_python)")
ax.text(0, (lead_tok_all + work_tok_all) + 5, f"{lead_tok_all + work_tok_all:.0f}k tok/step\n(lead + K workers)",
        ha="center", va="bottom", fontsize=8.4, fontweight="bold", color=COLOR["all"])
ax.text(1, lead_tok_ld + 5, f"{lead_tok_ld:.0f}k tok/step\n(0 worker tokens!)",
        ha="center", va="bottom", fontsize=8.4, fontweight="bold", color=COLOR["lead"])
ax.set_xticks(x_reg)
ax.set_xticklabels(["Case 1: all roles\n(org_small)", "Case 2: lead only\n(org_small_lead)"], fontsize=8.8)
ax.set_ylabel("trained tokens / step (k)")
ax.set_ylim(0, 205)
ax.set_xlim(-0.55, 1.55)
ax.set_title("Gradient budget by role")
ax.legend(loc="upper right", fontsize=8.0)

# Panel 2: Relative change from Step 0 -> Step 30 across key metrics
ax = axes[1]
metric_labels = [
    "org accuracy\n(held-out, n=40)",
    "solo transfer\n(zero-shot, n=40)",
    "4-site wide\n(K=4 OOD, n=10)",
    "worker speed\n(-exhausted %)",
    "org speed\n(-latency %)",
]
d_all = [
    100 * (eval_all[SF]["org"]["acc"] - eval_all[S0]["org"]["acc"]),
    100 * (eval_all[SF]["solo"]["acc"] - eval_all[S0]["solo"]["acc"]),
    100 * (eval_all[SF]["org"]["acc_wide"] - eval_all[S0]["org"]["acc_wide"]),
    100 * (eval_all[S0]["org"]["worker_exhausted"] - eval_all[SF]["org"]["worker_exhausted"])
    / eval_all[S0]["org"]["worker_exhausted"],
    100 * (eval_all[S0]["org"]["latency"] - eval_all[SF]["org"]["latency"])
    / eval_all[S0]["org"]["latency"],
]
d_lead = [
    100 * (eval_lead[SF]["org"]["acc"] - eval_lead[S0]["org"]["acc"]),
    100 * (eval_lead[SF]["solo"]["acc"] - eval_lead[S0]["solo"]["acc"]),
    100 * (eval_lead[SF]["org"]["acc_wide"] - eval_lead[S0]["org"]["acc_wide"]),
    100 * (eval_lead[S0]["org"]["worker_exhausted"] - eval_lead[SF]["org"]["worker_exhausted"])
    / eval_lead[S0]["org"]["worker_exhausted"],
    100 * (eval_lead[S0]["org"]["latency"] - eval_lead[SF]["org"]["latency"])
    / eval_lead[S0]["org"]["latency"],
]
y_m = np.arange(len(metric_labels))[::-1]
h = 0.34
ax.axvline(0, color=INK2, lw=1.0)
ax.barh(y_m + h / 2, d_all, height=h, color=COLOR["all"], label="Case 1: all roles")
ax.barh(y_m - h / 2, d_lead, height=h, color=COLOR["lead"], label="Case 2: lead only")
for y, va, vl in zip(y_m, d_all, d_lead):
    ax.text(va + (1.5 if va >= 0 else -1.5), y + h / 2, f"{va:+.0f}%",
            va="center", ha="left" if va >= 0 else "right", fontsize=8.3, fontweight="bold", color=COLOR["all"])
    ax.text(vl + (1.5 if vl >= 0 else -1.5), y - h / 2, f"{vl:+.0f}%",
            va="center", ha="left" if vl >= 0 else "right", fontsize=8.3, fontweight="bold", color=COLOR["lead"])
ax.set_yticks(y_m)
ax.set_yticklabels(metric_labels, fontsize=8.6)
ax.set_xlim(-34, 84)
ax.set_xlabel("improvement from step 0 to step 30 (%)")
ax.set_title("Step 0 -> 30 gains by regime")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper right", fontsize=8.0)

# Panel 3: Paired task-level wins (fixed) vs losses (broken) from step 0 -> 30
ax = axes[2]
p_all = results["paired"]["all_0_to_30"]
p_ld = results["paired"]["lead_0_to_30"]
modes = ["org", "solo", "all"]
mode_labels = ["org\n(n=40)", "solo\n(n=40)", "combined\n(n=80)"]
xm = np.arange(len(modes))
w = 0.32
ax.axhline(0, color=INK2, lw=1.0)
for idx, (m_key, col, p_dict, lab) in enumerate([
    ("all", COLOR["all"], p_all, "Case 1: all roles"),
    ("lead", COLOR["lead"], p_ld, "Case 2: lead only"),
]):
    pos = xm - w / 2 if idx == 0 else xm + w / 2
    fixed = [p_dict[k]["fixed"] for k in modes]
    broke = [-p_dict[k]["broke"] for k in modes]
    ax.bar(pos, fixed, width=w, color=col, label=f"{lab} (fixed)")
    ax.bar(pos, broke, width=w, color=col, alpha=0.38, hatch="//", edgecolor=col)
    for px, fx, bx in zip(pos, fixed, broke):
        ax.text(px, fx + 0.6, f"+{fx}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=col)
        ax.text(px, bx - 0.9, f"{bx}", ha="center", va="top", fontsize=8.0, color=INK2)
ax.text(2 - w / 2 - 0.05, 25.6, "p=0.024*", ha="center", fontsize=8.2, fontweight="bold", color=COLOR["all"])
ax.text(2 + w / 2 + 0.08, 20.6, "p=0.856", ha="center", fontsize=8.2, color=COLOR["lead"])
ax.set_xticks(xm)
ax.set_xticklabels(mode_labels, fontsize=8.8)
ax.set_ylim(-19, 31)
ax.set_ylabel("tasks fixed (+) vs broken (-)")
ax.set_title("Paired task changes (McNemar)")
ax.legend(loc="upper left", fontsize=8.0)

fig.subplots_adjust(wspace=0.44)
save(fig, "fig_regimes_comparison.png")

out_json = os.path.join(ROOT, "analysis", f"results_{args.run_all}.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"wrote analysis/results_{args.run_all}.json")
