"""Run organisations on tasks through the capture proxy, grade them, and turn every agent's captured
calls into training rows that all share the organisation's collective advantage."""
import asyncio
import uuid
from collections import defaultdict

import numpy as np

from agents import LEAD_MAX_TURNS, SOLO_MAX_TURNS, WORKER_MAX_TURNS, org_harness, solo_harness
from tasks import grade

EFF_BONUS = 0.1
# latency (model calls on the critical path) of the fastest plausible solve and of a slow one
LATENCY_RANGE = {"org": (4, LEAD_MAX_TURNS + 2 * WORKER_MAX_TURNS), "solo": (2, SOLO_MAX_TURNS)}


def reward_fn(correct, latency, mode, eff_bonus=EFF_BONUS):
    """Collective reward: 1 if the organisation's submitted answer is right, plus up to `eff_bonus` for a
    short critical path (asking sites in parallel beats asking them one by one). 0 otherwise."""
    if not correct:
        return 0.0
    lo, hi = LATENCY_RANGE[mode]
    return 1.0 + eff_bonus * min(1.0, max(0.0, (hi - latency) / (hi - lo)))


async def run_rollout(proxy, sbx, task, mode, eff_bonus=EFF_BONUS):
    rid = uuid.uuid4().hex
    url_for = lambda agent: proxy.base_url(f"{rid}-{agent}")
    if mode == "org":
        wss = {s: sbx.new_workspace(files) for s, files in task.sites.items()}
        run = org_harness(task, wss, url_for)
    else:
        wss = {"all": sbx.new_workspace({f"{s}/{fn}": text for s, fs in task.sites.items()
                                         for fn, text in fs.items()})}
        run = solo_harness(task, wss["all"], url_for)
    try:
        res = await run
    except Exception as e:   # a crashing harness is a failed rollout, not a crashed run
        agents = ["lead", *task.sites] if mode == "org" else ["solo"]
        res = {"answer": None, "status": f"error:{type(e).__name__}", "agents": agents, "latency": 0,
               "roles": {a: ("lead" if a == "lead" else "solo" if a == "solo" else "worker") for a in agents}}
    finally:
        for ws in wss.values():
            ws.cleanup()
    correct = grade(res["answer"], task)
    calls = {a: proxy.pop(f"{rid}-{a}") for a in res.pop("agents")}
    tok = defaultdict(int)
    for a, cs in calls.items():
        tok[res["roles"][a]] += sum(len(c["completion_ids"]) for c in cs)
    return {"task_id": task.task_id, "mode": mode, "difficulty": task.difficulty, "bucket": task.bucket,
            "template": task.template, "gold": task.answer, **res, "correct": bool(correct),
            "reward": reward_fn(correct, res.get("latency", 0), mode, eff_bonus),
            "gen_tokens": dict(tok), "calls": calls}


def tandem_repeat(ids, max_unit=32, min_repeats=20, min_span=300):
    """Avalon-M1's degenerate-loop test: a unit of <= max_unit tokens repeated >= min_repeats times
    covering >= min_span tokens."""
    if len(ids) < min_span:
        return False
    a = np.asarray(ids)
    for u in range(1, max_unit + 1):
        eq = np.concatenate([[0], (a[u:] == a[:-u]).astype(np.int8), [0]])
        d = np.diff(eq)
        runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
        if len(runs) and (r := runs.max()) >= (min_repeats - 1) * u and r + u >= min_span:
            return True
    return False


def to_rows(calls, role, agent):
    """One agent's calls -> training rows. Consecutive calls whose prompt extends the previous
    prompt+completion token for token are merged into one row (every harness here is append-only).
    Each row is (ids, loss_mask, old_logprobs); calls with a tandem-repeat loop get no loss."""
    rows, masked = [], 0
    for c in calls:
        p, comp, lp = c["prompt_ids"], c["completion_ids"], c["logprobs"]
        bit = 1
        if tandem_repeat(comp):
            bit, masked = 0, masked + 1
        if rows:
            ids = rows[-1]["ids"]
            if len(p) >= len(ids) and p[:len(ids)] == ids:
                ext = p[len(ids):]
                rows[-1]["ids"] += ext + comp
                rows[-1]["mask"] += [0] * len(ext) + [bit] * len(comp)
                rows[-1]["old_lp"] += [0.0] * len(ext) + lp
                continue
        rows.append({"ids": list(p) + comp, "mask": [0] * len(p) + [bit] * len(comp),
                     "old_lp": [0.0] * len(p) + lp, "role": role, "agent": agent})
    return [r for r in rows if any(r["mask"])], masked


def collective_samples(results, group, train_roles="all", std_floor=0.05):
    """GRPO over organisations with one collective advantage (Avalon-M1 RLCR): results are consecutive
    groups of `group` rollouts of the same task; every loss token of every agent in an organisation gets
    that organisation's group-normalised reward. Errored rollouts leave the group, zero-variance groups
    are dropped, and train_roles in {"lead", "worker"} keeps only that role's tokens."""
    samples, info = [], {"flat_groups": 0, "errors": 0, "loop_masked_calls": 0}
    for g in range(0, len(results), group):
        grp = [r for r in results[g:g + group] if not r["status"].startswith("error")]
        info["errors"] += len(results[g:g + group]) - len(grp)
        rs = [r["reward"] for r in grp]
        mu = sum(rs) / max(1, len(rs))
        sd = (sum((x - mu) ** 2 for x in rs) / max(1, len(rs))) ** 0.5
        if len(grp) < 2 or sd < 1e-6:
            info["flat_groups"] += 1
            continue
        for r in grp:
            a = (r["reward"] - mu) / (sd + std_floor)
            for agent, calls in r["calls"].items():
                role = r["roles"][agent]
                if train_roles != "all" and role != train_roles:
                    continue
                rows, masked = to_rows(calls, role, agent)
                info["loop_masked_calls"] += masked
                info[f"loss_tokens_{role}"] = info.get(f"loss_tokens_{role}", 0) + sum(sum(x["mask"]) for x in rows)
                samples += [(row, a) for row in rows]
    return samples, info


def summarize(rs):
    """Aggregate metrics over a list of rollout results (any mix of modes)."""
    out = {"n": len(rs), "acc": _mean(r["correct"] for r in rs), "latency": _mean(r["latency"] for r in rs),
           "submitted": _mean(r["status"] == "submitted" for r in rs),
           "statuses": dict(_count(r["status"] for r in rs)),
           "tokens": {k: _mean(r["gen_tokens"].get(k, 0) for r in rs) for k in ("lead", "worker", "solo")
                      if any(k in r["gen_tokens"] for r in rs)}}
    org = [r for r in rs if r["mode"] == "org"]
    if org:
        out.update({k: _mean(r.get(k, 0) for r in org) for k in
                    ("coverage", "max_fanout", "asks", "bad_asks", "lead_turns", "worker_calls",
                     "worker_tool_calls", "reply_fallbacks", "worker_exhausted")})
        out["full_coverage"] = _mean(r.get("coverage", 0) == 1 for r in org)
        out["acc_given_full_coverage"] = _mean(r["correct"] for r in org if r.get("coverage", 0) == 1)
    return out


async def evaluate(proxy, sbx, tasks, modes, temperature=0.7):
    old_t = proxy.temperature
    proxy.temperature = temperature
    results = await asyncio.gather(*[run_rollout(proxy, sbx, t, m) for t in tasks for m in modes])
    proxy.temperature = old_t
    out = {}
    for m in modes:
        rs = [r for r in results if r["mode"] == m]
        out[m] = summarize(rs)
        for b in sorted({r["bucket"] for r in rs}):
            out[m]["acc_" + b] = _mean(r["correct"] for r in rs if r["bucket"] == b)
        for k in sorted({r["n_sites"] for r in rs}):
            out[m][f"acc_k{k}"] = _mean(r["correct"] for r in rs if r["n_sites"] == k)
    for r in results:
        r.pop("calls")
    return out, results


def _mean(it):
    it = list(it)
    return sum(it) / max(1, len(it))


def _count(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return d
