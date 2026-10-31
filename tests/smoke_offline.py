"""Offline end-to-end smoke test (no GPU): a scripted stand-in for vLLM's /v1/completions drives real
organisations through the real capture proxy, Docker sandbox, grading and row building.

The scripted policy plays every role from the rendered prompt: the lead asks all sites in parallel for
their row count and submits the sum; workers run pandas and reply; solo sums over site directories.
So org and solo should both solve the 'how many rows' style question exactly, and every agent's calls
should merge into a single training row (token-level prompt extension).

    ../rl_harness/.venv-lfm/bin/python tests/smoke_offline.py
"""
import asyncio
import os
import re
import sys

from aiohttp import web
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tasks as T  # noqa: E402
from org_tasks import OrgTask, make_org_test, shard  # noqa: E402
from proxy import CaptureProxy  # noqa: E402
from rollout import collective_samples, run_rollout, tandem_repeat, to_rows  # noqa: E402
from sandbox import Sandbox  # noqa: E402

MODEL = "LiquidAI/LFM2.5-2.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
EOS = tok.convert_tokens_to_ids("<|im_end|>")


def call(name, **kw):
    return "<|tool_call_start|>[" + name + "(" + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + ")]<|tool_call_end|>"


def policy(prompt):
    """Scripted multi-role behaviour, chosen from the last turn of the rendered prompt."""
    turns = prompt.split("<|im_start|>")
    last = turns[-2]   # the message before the open assistant turn
    if "You lead a small analytics" in prompt:
        if last.startswith("user"):
            sites = re.search(r"Sites: (.*?)\.\n", prompt).group(1).split(", ")
            return "<|tool_call_start|>[" + ", ".join(
                f"ask(site={s!r}, request='How many rows are in your main table?')" for s in sites) + "]<|tool_call_end|>"
        n = sum(int(x) for x in re.findall(r"\[\w+\] (\d+)", prompt))
        return call("submit", answer=str(n))
    if "You are the data specialist" in prompt:
        main = re.search(r"files only: (\w+\.csv)", prompt).group(1)
        if last.startswith("user"):
            return call("run_python", code=f"import pandas as pd\nprint(len(pd.read_csv('{main}')))")
        return call("reply", message=re.findall(r"\d+", last)[-1])
    if last.startswith("user"):
        return call("run_python", code="import glob, pandas as pd\nprint(sum(len(pd.read_csv(f)) for f in "
                                       "glob.glob('*/*.csv') if 'customers' not in f and 'departments' not in f "
                                       "and 'users' not in f))")
    return call("final_answer", answer=re.findall(r"\d+", last)[-1])


async def completions(request):
    body = await request.json()
    prompt = tok.decode(body["prompt"])
    text = "Let me think.</think>" if prompt.rstrip().endswith("<think>") else "<think>Let me think.</think>"
    ids = tok.encode(text + policy(prompt), add_special_tokens=False) + [EOS]
    return web.json_response({"choices": [{"text": tok.decode(ids), "finish_reason": "stop", "logprobs": {
        "tokens": [f"token_id:{i}" for i in ids], "token_logprobs": [-0.1] * len(ids)}}]})


def count_task(seed, k):
    """A task whose answer is the main table's row count (what the scripted policy computes)."""
    import random
    rng = random.Random(seed)
    x = T.make_retail(rng)
    files, _, _ = T._messy_files(rng, x, "easy")
    names = ["atlas", "boreal", "cobalt", "delta"][:k]
    parts = shard(rng, files[x.main], names)
    return OrgTask(task_id=f"c{seed}", question="How many orders are there in total?", answer=str(len(x.o)),
                   kind="number", difficulty="easy", files=files, template="count", main=x.main,
                   shared=["customers.csv"],
                   sites={n: {x.main: parts[n], "customers.csv": files["customers.csv"]} for n in names})


async def main():
    app = web.Application(client_max_size=64 * 2**20)
    app.router.add_post("/v1/completions", completions)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 8199).start()
    proxy = CaptureProxy(tok, vllm_url="http://127.0.0.1:8199", model_name=MODEL, max_tokens=512, max_ctx=16384,
                         port=8198)
    await proxy.start()
    sbx = Sandbox(name="sbx-smoke")
    try:
        tasks = [count_task(s, k) for s, k in [(1, 2), (2, 3), (3, 4)]]
        results = await asyncio.gather(*[run_rollout(proxy, sbx, t, m) for t in tasks for m in ("org", "solo")])
        for r in results:
            calls = r["calls"]
            rows = {a: to_rows(cs, r["roles"][a], a)[0] for a, cs in calls.items()}
            print(f"{r['mode']:4s} k={r['n_sites']} gold={r['gold']} ans={r['answer']} {r['status']} "
                  f"correct={r['correct']} reward={r['reward']:.3f} latency={r['latency']} cov={r.get('coverage')} "
                  f"fanout={r.get('max_fanout')} calls={ {a: len(c) for a, c in calls.items()} } "
                  f"rows={ {a: len(x) for a, x in rows.items()} }")
            assert r["correct"], r
            assert all(len(x) == 1 for x in rows.values()), "every append-only agent should give one row"
            for x in rows.values():   # loss tokens are exactly the sampled tokens
                row = x[0]
                sampled = [i for i, m in zip(row["ids"], row["mask"]) if m]
                assert sampled == [t for c in calls[row["agent"]] for t in c["completion_ids"]]
        print(f"proxy: {proxy.n_extended}/{proxy.n_calls} calls extended token-for-token")
        org = [r for r in results if r["mode"] == "org"]
        assert all(r["latency"] == 4 and r["coverage"] == 1 for r in org)
        assert tandem_repeat([5, 6, 7] * 120) and not tandem_repeat(list(range(400)))
        assert len(make_org_test(2)) == 8
        # collective advantage: group of 3 orgs on one task, one wrong -> every agent's rows share its sign
        grp = [r for r in results if r["mode"] == "org"]
        grp[0] = dict(grp[0], reward=0.0)
        samples, info = collective_samples(grp, 3)
        assert len(samples) == sum(len(r["calls"]) for r in grp) and info["flat_groups"] == 0, info
        n_wrong = len(grp[0]["calls"])   # rows of the failed org: all negative, all others positive
        assert sum(a < 0 for _, a in samples) == n_wrong and len({a for _, a in samples}) == 2
        lead_only, _ = collective_samples(grp, 3, "lead")
        assert {row["role"] for row, _ in lead_only} == {"lead"} and len(lead_only) == 3
        flat, info = collective_samples([dict(r, reward=1.0) for r in grp], 3)
        assert not flat and info["flat_groups"] == 1
        print("collective_samples", info)
        print("OK")
    finally:
        await proxy.stop()
        await runner.cleanup()
        sbx.close()


asyncio.run(main())
