"""Extra Results-section figures for the multi-agent RL blog post (complements make_blog_figures.py).

    python analysis/make_results_figures.py --out blog/multi-agent-rl

Writes:
  fig_inside_team.png  step-30 coordination metrics relative to the base model, lead-only vs all roles
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--run-all", default="org_small")
p.add_argument("--run-lead", default="org_small_lead")
p.add_argument("--out", default=os.path.join(ROOT, "blog", "multi-agent-rl"))
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

# Same palette and style as make_blog_figures.py
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


ev_all = {e["step"]: e for e in jl(args.run_all, "eval.jsonl")}
ev_lead = {e["step"]: e for e in jl(args.run_lead, "eval.jsonl")}
S0, SF = min(ev_all), max(ev_all)


def save(fig, name):
    fig.savefig(os.path.join(args.out, name))
    plt.close(fig)
    print("wrote", name)



# ------------------------------------------------------------- fig_inside_team: coordination metrics vs base
b, ld, al = ev_all[S0], ev_lead[SF], ev_all[SF]
mt = lambda e, mode: e[mode]["statuses"].get("max_turns", 0)
metrics = [  # (label, base, lead-only step 30, all roles step 30, format)
    ("Critical-path latency (calls)", b["org"]["latency"], ld["org"]["latency"], al["org"]["latency"], "{:.1f}"),
    ("Workers hitting the 6-call cap / task", b["org"]["worker_exhausted"], ld["org"]["worker_exhausted"],
     al["org"]["worker_exhausted"], "{:.2f}"),
    ("Worker model calls / task", b["org"]["worker_calls"], ld["org"]["worker_calls"], al["org"]["worker_calls"],
     "{:.1f}"),
    ("Worker sandbox tool calls / task", b["org"]["worker_tool_calls"], ld["org"]["worker_tool_calls"],
     al["org"]["worker_tool_calls"], "{:.1f}"),
    ("Worker plain-text reply fallbacks / task", b["org"]["reply_fallbacks"], ld["org"]["reply_fallbacks"],
     al["org"]["reply_fallbacks"], "{:.2f}"),
    ("Worker tokens / task", b["org"]["tokens"]["worker"], ld["org"]["tokens"]["worker"],
     al["org"]["tokens"]["worker"], "{:,.0f}"),
    ("Lead tokens / task", b["org"]["tokens"]["lead"], ld["org"]["tokens"]["lead"], al["org"]["tokens"]["lead"],
     "{:,.0f}"),
    ("Org episodes timing out (of 40)", mt(b, "org"), mt(ld, "org"), mt(al, "org"), "{:.0f}"),
    ("Solo episodes timing out (of 40)", mt(b, "solo"), mt(ld, "solo"), mt(al, "solo"), "{:.0f}"),
    ("Eval wall-clock, 80 rollouts (s)", b["time"], ld["time"], al["time"], "{:.0f}"),
]
fig, ax = plt.subplots(figsize=(9.6, 5.2))
y = np.arange(len(metrics))[::-1]
ax.axvline(100, color=INK2, lw=1.0)
for yi, (lab, vb, vl, va, fmt) in zip(y, metrics):
    rl, ra = 100 * vl / vb, 100 * va / vb
    ax.plot([min(rl, ra), max(rl, ra)], [yi, yi], color=BASE, lw=2.5, zorder=1)
    ax.scatter([rl], [yi], s=70, color=COLOR["lead"], zorder=3)
    ax.scatter([ra], [yi], s=70, color=COLOR["all"], zorder=3)
    tr = ax.get_yaxis_transform()   # x in axes fraction, y in data
    for xf, v, col, wt in ((1.04, vb, INK2, "normal"), (1.17, vl, COLOR["lead"], "bold"),
                           (1.30, va, COLOR["all"], "bold")):
        ax.text(xf, yi, fmt.format(v), transform=tr, va="center", ha="left",
                fontsize=8.8, color=col, fontweight=wt)
tr = ax.get_yaxis_transform()
for xf, head, col in ((1.04, "base", INK2), (1.17, "lead only", COLOR["lead"]), (1.30, "all roles", COLOR["all"])):
    ax.text(xf, len(metrics) - 0.35, head, transform=tr, va="bottom", ha="left", fontsize=8.8,
            fontweight="bold", color=col)
ax.set_yticks(y)
ax.set_yticklabels([m[0] for m in metrics], fontsize=9.2)
ax.set_xlim(10, 120)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("step-30 value as % of the base model (lower is better on every row)")
ax.set_title("What changed inside the organisation after 30 steps")
ax.grid(axis="y", visible=False)
ax.scatter([], [], s=70, color=COLOR["lead"], label="lead only (step 30)")
ax.scatter([], [], s=70, color=COLOR["all"], label="all roles (step 30)")
ax.plot([], [], color=INK2, lw=1.0, label="base model (step 0) = 100%")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8.8)
save(fig, "fig_inside_team.png")
