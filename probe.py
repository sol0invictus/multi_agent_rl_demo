"""Base-model probe: run organisations (and the solo baseline) on test tasks without training and print
per-bucket accuracy and coordination stats; with --show, print a few full organisation transcripts."""
import argparse
import asyncio
import json
from collections import Counter

from transformers import AutoTokenizer

from org_tasks import TEST_BUCKETS, make_org_test
from proxy import CaptureProxy
from rollout import evaluate, run_rollout
from sandbox import Sandbox

p = argparse.ArgumentParser()
p.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
p.add_argument("--vllm", default="http://127.0.0.1:8000")
p.add_argument("--modes", default="org,solo")
p.add_argument("--buckets", default=",".join(TEST_BUCKETS))
p.add_argument("--n", type=int, default=10, help="tasks per bucket")
p.add_argument("--out", default="")
p.add_argument("--show", type=int, default=0, help="print this many org transcripts")
p.add_argument("--port", type=int, default=8101)
args = p.parse_args()


def show(tok, r):
    print(f"\n===== {r['template']} k={r['n_sites']} gold={r['gold']} -> {r['answer']!r} {r['status']} "
          f"correct={r['correct']} latency={r['latency']} coverage={r.get('coverage')}")
    for agent, calls in r["calls"].items():
        for i, c in enumerate(calls):
            print(f"--- [{agent} call {i}] {tok.decode(c['completion_ids'])[-1200:]}")


async def main():
    tok = AutoTokenizer.from_pretrained(args.model)
    proxy = CaptureProxy(tok, vllm_url=args.vllm, model_name=args.model, max_tokens=4096, max_ctx=16384,
                         port=args.port)
    await proxy.start()
    sbx = Sandbox(name="sbx-probe", max_concurrent=32)
    tasks = make_org_test(args.n, buckets=args.buckets.split(","))
    try:
        if args.show:
            proxy.temperature = 0.7
            for t in tasks[:args.show]:
                show(tok, await run_rollout(proxy, sbx, t, "org"))
        else:
            res, results = await evaluate(proxy, sbx, tasks, args.modes.split(","))
            print(json.dumps(res, indent=1))
            if args.out:
                with open(args.out, "w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
            for m in args.modes.split(","):
                print(m, Counter(r["status"] for r in results if r["mode"] == m))
    finally:
        await proxy.stop()
        sbx.close()


asyncio.run(main())
