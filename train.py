"""Multi-agent RL with LoRA on one GPU: reinforcement learning from collective rewards (Avalon-M1's RLCR,
scaled down to a 2.6B model and a single 3090 Ti).

Loop per step: sample partitioned-data tasks -> run a GRPO group of whole organisations per task (a lead
plus one worker per site, all the same policy) through the capture proxy -> score each organisation's
submitted answer -> give every sampled token of every agent in the organisation the same group-relative
advantage -> policy-gradient update on the exact captured tokens -> push merged weights to vLLM.
vLLM (separate process, serve_vllm.sh) and the trainer share the GPU and take turns.
"""
import argparse
import asyncio
import importlib.util
import json
import math
import os
import random
import time
from collections import defaultdict

import aiohttp
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from proxy import CaptureProxy
from org_tasks import TEST_BUCKETS, make_org_split, make_org_test
from rollout import collective_samples, evaluate, run_rollout, summarize
from sandbox import Sandbox

p = argparse.ArgumentParser()
p.add_argument("--run", default="org")
p.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
p.add_argument("--mode", default="org", choices=["org", "solo"], help="what is trained: organisations or one agent")
p.add_argument("--eval-modes", default="org,solo")
p.add_argument("--train-roles", default="all", choices=["all", "lead", "worker"],
               help="whose tokens get the collective advantage: all agents (Avalon RLCR) or only the lead "
                    "(Kimi PARL freezes sub-agents) / only the workers")
p.add_argument("--eff-bonus", type=float, default=0.1, help="max reward bonus for a short critical path")
p.add_argument("--steps", type=int, default=60)
p.add_argument("--tasks-per-step", type=int, default=8)
p.add_argument("--group", type=int, default=8)
p.add_argument("--lr", type=float, default=4e-5)
p.add_argument("--rank", type=int, default=32)
p.add_argument("--n-train", type=int, default=2000)
p.add_argument("--n-test", type=int, default=30, help="test tasks per bucket")
p.add_argument("--eval-buckets", default=",".join(TEST_BUCKETS), help="subset of seen,tmpl,domain,wide")
p.add_argument("--eval-every", type=int, default=10)
p.add_argument("--max-pack", type=int, default=8192)
p.add_argument("--max-tokens", type=int, default=4096, help="output cap per model call (blog: 4096)")
p.add_argument("--max-ctx", type=int, default=16384)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--eval-temperature", type=float, default=0.7)
p.add_argument("--is-cap", type=float, default=2.0, help="truncated importance-sampling cap")
p.add_argument("--adv-std-floor", type=float, default=0.05)
p.add_argument("--vllm", default="http://127.0.0.1:8000")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume-step", type=int, default=0,
               help="continue a crashed run from adapters/step_N (optimizer state too if optimizer.pt exists)")
p.add_argument("--sync", choices=["lora", "merge"], default="merge",
               help="push the policy to vLLM as a LoRA adapter or as merged full weights")
args = p.parse_args()

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.run)
os.makedirs(RUN_DIR, exist_ok=True)
random.seed(args.seed)
torch.manual_seed(args.seed)


def log(name, obj):
    with open(os.path.join(RUN_DIR, name), "a") as f:
        f.write(json.dumps(obj) + "\n")


# ------------------------------------------------------------------------------------- policy
tok = AutoTokenizer.from_pretrained(args.model)
ATTN = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                             attn_implementation=ATTN).cuda()
# LFM2's short convolutions would leak state across rows packed into one sequence, so pack only
# pure-attention models; LFM2 trains one row per forward.
PACK = ATTN == "flash_attention_2" and "conv" not in (getattr(model.config, "layer_types", None) or [])
if model.config.model_type == "lfm2":    # attention + MLP (vLLM serves these as qkv_proj / w13 / w2)
    TARGETS = r".*\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|feed_forward\.(w1|w2|w3))"
else:
    TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=TARGETS))
model.print_trainable_parameters()
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr,
                        betas=(0.9, 0.99), weight_decay=0.0)
backbone = model.base_model.model.model          # LoRA layers are injected in place
lm_head_w = model.base_model.model.lm_head.weight


def _chunk_logprobs(h, targets):
    logits = (h @ lm_head_w.T).float()
    return logits.log_softmax(-1).gather(-1, targets[:, None])[:, 0]


def packed_logprobs(rows):
    """Log-probs of every loss token in a pack of rows (flash-attn varlen via position_ids resets)."""
    ids = torch.tensor([t for r in rows for t in r["ids"]], device="cuda")[None]
    pos = torch.tensor([i for r in rows for i in range(len(r["ids"]))], device="cuda")[None]
    hidden = backbone(input_ids=ids, position_ids=pos).last_hidden_state[0]
    # predict token t from position t-1; loss positions are where mask==1 (never the first token of a row)
    sel, off = [], 0
    for r in rows:
        sel += [off + i - 1 for i, m in enumerate(r["mask"]) if m]
        off += len(r["ids"])
    sel = torch.tensor(sel, device="cuda")
    h, tgt = hidden[sel], ids[0, sel + 1]
    out = [checkpoint(_chunk_logprobs, h[i:i + 2048], tgt[i:i + 2048], use_reentrant=False)
           for i in range(0, len(sel), 2048)]
    return torch.cat(out)


def train_step(samples):
    """samples: list of (row, advantage). One optimizer step, token-level mean over the batch."""
    model.train()
    n_tok = sum(sum(r["mask"]) for r, _ in samples)
    packs, cur, cur_len = [], [], 0
    for s in sorted(samples, key=lambda s: -len(s[0]["ids"])):
        L = len(s[0]["ids"])
        if cur and (not PACK or cur_len + L > args.max_pack):
            packs.append(cur)
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += L
    if cur:
        packs.append(cur)
    stats = defaultdict(float)
    for pack in packs:
        rows = [r for r, _ in pack]
        lp = packed_logprobs(rows)
        old = torch.tensor([x for r in rows for x, m in zip(r["old_lp"], r["mask"]) if m], device="cuda")
        adv = torch.tensor([a for r, a in pack for m in r["mask"] if m], device="cuda")
        log_ratio = lp.detach() - old                     # trainer vs sampler (vLLM) mismatch
        w = log_ratio.exp().clamp(max=args.is_cap)
        loss = -(w * adv * lp).sum() / n_tok
        loss.backward()
        stats["loss"] += loss.item()
        stats["mismatch_kl"] += (-log_ratio).sum().item() / n_tok   # E_sampler[log q - log p]
        stats["is_clipped"] += (log_ratio.exp() > args.is_cap).sum().item() / n_tok
        stats["entropy_proxy"] += (-lp.detach()).sum().item() / n_tok
    gn = torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    stats.update(grad_norm=gn.item(), train_tokens=n_tok, packs=len(packs),
                 peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                 seq_tokens=sum(len(r["ids"]) for r, _ in samples))
    return dict(stats)


async def publish_adapter(session, step, prev_name):
    path = os.path.join(RUN_DIR, "adapters", f"step_{step}")
    model.save_pretrained(path)
    name = f"{args.run}_s{step}"
    for attempt in range(5):   # vLLM can stall on the load while it drains requests; don't kill the run
        try:
            body = {"lora_name": name, "lora_path": path, "load_inplace": True}   # idempotent across retries
            async with session.post(f"{args.vllm}/v1/load_lora_adapter", json=body,
                                    timeout=aiohttp.ClientTimeout(total=600)) as r:
                assert r.status == 200, await r.text()
            break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"load_lora_adapter attempt {attempt + 1} failed: {e!r}", flush=True)
            if attempt == 4:
                raise
            await asyncio.sleep(30)
    if prev_name != args.model:
        async with session.post(f"{args.vllm}/v1/unload_lora_adapter", json={"lora_name": prev_name}) as r:
            pass
    return name


@torch.no_grad()
def merged_state_dict():
    """Base weights with W + scale * B @ A folded in (fp32 math, base weights untouched), in HF names."""
    sd = {}
    for name, mod in model.base_model.model.named_modules():
        if hasattr(mod, "lora_A") and "default" in mod.lora_A:
            delta = mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float()
            w = mod.base_layer.weight.float() + mod.scaling["default"] * delta
            sd[name + ".weight"] = w.to(torch.bfloat16).cpu()
    for name, prm in model.base_model.model.named_parameters():
        name = name.replace(".base_layer", "")
        if "lora_" in name or name in sd or (name == "lm_head.weight" and model.config.tie_word_embeddings):
            continue
        sd[name] = prm.detach().cpu()
    return sd


async def publish_merged(session, step):
    """Sync the policy into vLLM by reloading full merged weights in place. vLLM's LoRA path drops out of
    CUDA graphs for LFM2 and decodes ~15x slower on this box, so the server runs without LoRA."""
    if step > 0:
        model.save_pretrained(os.path.join(RUN_DIR, "adapters", f"step_{step}"))   # small checkpoint to keep
        torch.save(opt.state_dict(), os.path.join(RUN_DIR, "optimizer.pt"))         # latest only, for --resume-step
    path = os.path.join(RUN_DIR, "policy")
    os.makedirs(path, exist_ok=True)
    save_file(merged_state_dict(), os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    rpc = {"method": "reload_weights", "kwargs": {"weights_path": path}}
    for attempt in range(5):   # one connect stall killed a run at step 21; reloading is idempotent, so retry
        try:
            async with session.post(f"{args.vllm}/collective_rpc", json=rpc,
                                    timeout=aiohttp.ClientTimeout(total=600, sock_connect=60)) as r:
                assert r.status == 200, await r.text()
            async with session.post(f"{args.vllm}/reset_prefix_cache",   # cached KV came from the old weights
                                    timeout=aiohttp.ClientTimeout(total=120, sock_connect=60)) as r:
                assert r.status == 200, await r.text()
            return
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"weight sync attempt {attempt + 1} failed: {e!r}", flush=True)
            if attempt == 4:
                raise
            await asyncio.sleep(30)


# --------------------------------------------------------------------------------------- main
async def main():
    train_tasks = make_org_split(args.n_train, 21)
    test_tasks = make_org_test(args.n_test, buckets=args.eval_buckets.split(","))
    eval_modes = args.eval_modes.split(",")
    template_kwargs = {"enable_thinking": False} if "Qwen3" in args.model else {}
    proxy = CaptureProxy(tok, vllm_url=args.vllm, model_name=args.model, temperature=args.temperature,
                         max_tokens=args.max_tokens, max_ctx=args.max_ctx, template_kwargs=template_kwargs)
    await proxy.start()
    sbx = Sandbox(name=f"sbx-{args.run}", max_concurrent=32)
    session = aiohttp.ClientSession()
    log("config.json", vars(args))
    start = args.resume_step + 1
    if args.resume_step:
        adir = os.path.join(RUN_DIR, "adapters", f"step_{args.resume_step}")
        res = set_peft_model_state_dict(model, load_file(os.path.join(adir, "adapter_model.safetensors"), device="cuda"))
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        opt_path = os.path.join(RUN_DIR, "optimizer.pt")
        if os.path.exists(opt_path):
            opt.load_state_dict(torch.load(opt_path))
        else:
            print("resume: no optimizer.pt, AdamW moments restart from zero", flush=True)
        if args.sync == "merge":
            await publish_merged(session, 0)
        else:
            proxy.model_name = await publish_adapter(session, args.resume_step, proxy.model_name)

    async def do_eval(step):
        t0 = time.time()
        res, results = await evaluate(proxy, sbx, test_tasks, eval_modes, args.eval_temperature)
        res.update(step=step, time=time.time() - t0, model=proxy.model_name)
        log("eval.jsonl", res)
        log("eval_rollouts.jsonl", {"step": step, "results": results})
        bk = lambda d: "/".join(f"{d.get('acc_' + b, float('nan')):.2f}" for b in TEST_BUCKETS)
        line = []
        for m in eval_modes:
            d = res[m]
            line.append(f"{m}={d['acc']:.3f} [{bk(d)}] lat={d['latency']:.1f}"
                        + (f" cov={d['coverage']:.2f} fan={d['max_fanout']:.2f}" if m == "org" else ""))
        print(f"[eval step {step}] ({'/'.join(TEST_BUCKETS)}) " + "  ".join(line) + f"  ({res['time']:.0f}s)",
              flush=True)

    if not args.resume_step:
        await do_eval(0)
    cursor = args.resume_step * args.tasks_per_step
    for step in range(start, args.steps + 1):
        t0 = time.time()
        batch = train_tasks[cursor:cursor + args.tasks_per_step]
        cursor += args.tasks_per_step
        jobs = [run_rollout(proxy, sbx, task, args.mode, args.eff_bonus) for task in batch for _ in range(args.group)]
        results = await asyncio.gather(*jobs)
        t_roll = time.time() - t0

        samples, info = collective_samples(results, args.group, args.train_roles, args.adv_std_floor)
        t1 = time.time()
        stats = train_step(samples) if samples else {}
        torch.cuda.empty_cache()
        if args.sync == "merge":
            await publish_merged(session, step)
        else:
            proxy.model_name = await publish_adapter(session, step, proxy.model_name)
        summ = summarize(results)
        m = {"step": step, "reward": sum(r["reward"] for r in results) / len(results),
             **{k: v for k, v in summ.items() if k not in ("statuses", "tokens", "n")},
             **info, "rows": len(samples),
             **{f"gen_tokens_{k}": v for k, v in summ["tokens"].items()},
             **{f"acc_k{k}": sum(r["correct"] for r in results if r["n_sites"] == k)
                / max(1, sum(r["n_sites"] == k for r in results)) for k in sorted({r["n_sites"] for r in results})},
             "t_rollout": t_roll, "t_train": time.time() - t1, **stats}
        log("metrics.jsonl", m)
        print(f"step {step:3d} reward={m['reward']:.3f} acc={m['acc']:.3f} lat={m['latency']:.1f} "
              + (f"cov={m['coverage']:.2f} fan={m['max_fanout']:.2f} asks={m['asks']:.1f} " if args.mode == "org" else "")
              + f"flat={info['flat_groups']}/{len(batch)} err={info['errors']} rows={len(samples)} tok={stats.get('seq_tokens', 0)} "
              f"kl={stats.get('mismatch_kl', 0):.4f} gn={stats.get('grad_norm', 0):.3f} "
              f"roll={t_roll:.0f}s train={m['t_train']:.0f}s mem={stats.get('peak_mem_gb', 0):.1f}G", flush=True)
        for r in results:
            r.pop("calls")
        log("train_rollouts.jsonl", {"step": step, "results": results})
        if step % args.eval_every == 0 or step == args.steps:
            await do_eval(step)
    await session.close()
    await proxy.stop()
    sbx.close()


asyncio.run(main())
