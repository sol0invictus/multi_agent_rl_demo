# multi_agent_rl: training a 2.6B model to work as an agent organisation

Multi-agent RL with LoRA on one RTX 3090 Ti. This is a scaled-down version of **reinforcement learning from
collective rewards (RLCR)** from [Avalon-M1](https://www.gr.inc/avalonm1.pdf), built on the same stack as
`../rl_harness`: LiquidAI/LFM2.5-2.6B, LoRA r32 on attention+MLP, GRPO with truncated importance sampling,
a token-capturing proxy in front of vLLM, merged-weight sync, and a Docker sandbox.

## The environment: a data organisation

Each task is a data question from `tasks.py` (retail / hr / web CSVs, 46 templates, messy-file variants).
The main table is **sharded row-wise across K sites** with uneven shares. Lookup tables are copied to every site.

| agent | sees | actions |
|---|---|---|
| **lead** (1) | the question and the site layout, **no data** | `ask(site, request)` (several in one turn run in parallel), `submit(answer)` |
| **worker** (one per site) | only its own site's files; keeps its conversation across requests | `list_files`, `read_file`, `run_python`, `reply(message)` |
| **solo** (baseline) | all sites' files in one workspace (`site/<file>`) | data tools + `final_answer` |

No agent can answer alone, so communication has a direct route to the reward. The lead must decompose the
question, ask every site for the right partial results and combine them. For example, a global average needs
per-site sums and counts, not per-site averages. A top-k needs per-group partial sums, and a distinct count
needs the IDs themselves. Workers must answer exactly what was asked.

All agents are **one shared policy**. Roles differ only in context (system prompt, tools, history), as in
Avalon ("θ functions like a gene pool").

## RL recipe (what is taken from where)

| piece | here | source |
|---|---|---|
| collective reward | one scalar per organisation: 1 if the submitted answer is right, plus up to 0.1 for a short critical path | Avalon RLCR |
| credit assignment | **every sampled token of every agent** gets the same group-normalised advantage; no per-agent credit | Avalon RLCR; team-reward PPO works in coop MARL (MAPPO, Yu et al. 2022) |
| groups | G whole organisations per task; zero-variance groups dropped; harness errors removed from the group | Avalon (K=16, drop flat groups) |
| latency incentive | reward bonus on critical-path model calls (parallel asks cost the slowest site), plus hard turn caps | Avalon's wall-clock cap / derived latency |
| loop masking | an agent's call gets no loss if a ≤32-token unit repeats ≥20 times over ≥300 tokens | Avalon tandem-repeat masking |
| ablation `--train-roles lead` | only the lead's tokens are trained; workers stay frozen at their behaviour | Kimi K2.5 PARL (trains the orchestrator only) |
| objective / infra | token-mean policy gradient with IS cap 2.0 against vLLM logprobs, LoRA, merged-weight reload | unchanged from `rl_harness` |

Not adopted: SPIRAL's role-conditioned advantage ([arXiv 2506.24119](https://arxiv.org/abs/2506.24119)). It
is for zero-sum self-play, where roles have opposite rewards. Here every agent shares the reward, so a
per-role baseline would equal the group baseline. It is the natural next step if a competitive environment
is added.

## Evaluation

`make_org_test(n)` builds four buckets, all with seeds disjoint from training:

- `seen`: training templates on fresh data, 2–3 sites
- `tmpl`: held-out question templates, 2–3 sites
- `domain`: held-out web domain, 2–3 sites
- `wide`: **4 sites**, more workers than training ever had (training uses 2–3)

Every eval runs the **org** and **solo** modes on the same tasks. Logged coordination metrics:

- `coverage`: share of sites asked
- `full_coverage` and `acc_given_full_coverage`: separate "forgot a site" from "combined the results wrong"
- `max_fanout`: sites asked in parallel in one turn
- `latency`, `asks`, `bad_asks`, `lead_turns`
- `worker_tool_calls`, `reply_fallbacks`: worker answered in plain text instead of calling `reply`
- `worker_exhausted`
- generated tokens per role

## Files

| file | what |
|---|---|
| `org_tasks.py` | partitioned tasks (sharding, test buckets); `python org_tasks.py` self-checks that shards reassemble exactly |
| `agents.py` | `org_harness` (lead + workers) and `solo_harness` |
| `rollout.py` | rollouts, collective reward, `collective_samples` (RLCR advantages), tandem-repeat masking, eval summaries |
| `train.py` | the `rl_harness` trainer with the multi-agent loop |
| `probe.py` | base-model eval without training; `--show N` prints full organisation transcripts |
| `tests/smoke_offline.py` | no-GPU end-to-end test: a scripted fake vLLM drives real orgs through proxy, sandbox, grading, rows and advantages |
| `analysis/summarize_run.py` | markdown tables for a run (eval per bucket, training curve) and `--compare` paired McNemar test between runs |
| `scripts/stop_vllm.sh` | stops the vLLM server by pid (`pkill -f` would kill the calling shell) |
| `Makefile`, `CLAUDE.md`, `requirements.txt` | common commands, agent/contributor guide, pinned versions of the shared venv |
| `proxy.py`, `sandbox.py`, `tasks.py`, `serve_vllm.sh`, `sandbox_image/` | copied from `rl_harness`. Small changes: sandbox subdirectories, `exclude=` for messy-file kinds, venv path |

## Running

Shortcuts are in the `Makefile` (`make test`, `make serve`, `make probe`, `make train RUN=...`,
`make ablation`, `make table RUN=...`). The equivalent raw commands:

```bash
# venv is shared with rl_harness (vLLM 0.32 / transformers 5.19); the sandbox image is harness-sandbox:latest
PY=../rl_harness/.venv-lfm/bin/python
$PY tests/smoke_offline.py                          # no GPU needed
nohup ./serve_vllm.sh > runs/vllm.log 2>&1 &        # ~45% of the GPU
$PY probe.py --n 10                                 # base org vs solo, all buckets
$PY probe.py --show 3                               # read what the lead and workers actually say
nohup $PY train.py --run org_all --steps 60 --eval-every 20 > runs/org_all.log 2>&1 &
# ablation: same seed, only the lead is trained
nohup $PY train.py --run org_lead --train-roles lead --steps 60 --eval-every 20 > runs/org_lead.log 2>&1 &
```

Defaults: 8 tasks × group 8 = 64 organisations per step, lr 4e-5, rank 32, 30 test tasks per bucket.
