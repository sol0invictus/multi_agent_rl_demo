# CLAUDE.md

Multi-agent RL (Avalon-M1-style collective rewards) on LiquidAI/LFM2.5-2.6B with LoRA + GRPO, on one RTX 3090 Ti
under WSL2. A lead agent with no data access coordinates per-site worker agents over sharded CSV data. All agents
are one shared policy. Design and rationale: `README.md`. **Current state and next steps: `STATUS.md`. Read it
first when resuming, and update it at the end of a session** (results tables, what is running, open items).

Sister project: `../rl_harness` (single-agent multi-harness RL). `proxy.py`, `sandbox.py`, `tasks.py`,
`serve_vllm.sh` and most of `train.py` were copied from it. Keep fixes in sync when they apply to both.

## Commands

```bash
make test        # no GPU: task self-check + offline end-to-end smoke test (scripted fake vLLM, real Docker)
make serve       # vLLM policy server on :8000, ~45% of GPU memory (log: runs/vllm.log)
make stop-vllm   # kill it safely (see gotchas)
make probe       # base-model org vs solo eval, 10 tasks per bucket
make show        # print 3 full organisation transcripts
make train RUN=org_all [ARGS="--steps 60"]          # background; log runs/$RUN.log
make table RUN=org_all                               # markdown results table from runs/$RUN/
```

Python is `../rl_harness/.venv-lfm/bin/python` (vLLM 0.32, transformers 5.19, torch 2.13 cu130, see
`requirements.txt`). There is no venv in this directory. The Docker image `harness-sandbox:latest` must exist
(`make image`).

## Layout

- `org_tasks.py`: shards a `tasks.py` question's main table across sites; test buckets seen/tmpl/domain/wide.
- `agents.py`: `org_harness` (lead + `Worker`s) and `solo_harness`. Harnesses only see an OpenAI URL.
- `rollout.py`: `run_rollout`, `reward_fn`, `to_rows`, `tandem_repeat`, `collective_samples` (RLCR advantages),
  `summarize` / `evaluate`.
- `train.py`: policy, LoRA, `train_step`, weight sync to vLLM, main loop. It loads the model on import, so
  testable logic belongs in `rollout.py`.
- `proxy.py`: capture proxy. Each agent gets its own rollout id `<rid>-<agent>`.
- `runs/<run>/`: `config.json`, `metrics.jsonl` (per step), `eval.jsonl`, `eval_rollouts.jsonl`,
  `train_rollouts.jsonl`, `adapters/step_N/`, `policy/` (latest merged weights), `optimizer.pt`.

## Invariants (do not break)

- **Train only on tokens the model sampled.** The proxy records prompt ids, completion ids and vLLM logprobs.
  Never re-tokenize model text for training. Prompts, tool outputs and messages from other agents get mask 0.
- **One collective advantage per organisation.** Every loss token of every agent gets the same group-normalised
  reward. Per-agent credit is an explicit ablation (`--train-roles`), never a silent change.
- **The lead has no data access, and workers see only their own site.** Otherwise communication stops mattering.
  Gold answers are always computed on the full clean data.
- **Test seeds and held-out templates/domains stay disjoint from training** (`python org_tasks.py` asserts it).
- Harnesses are black boxes to the trainer: they return a result dict, and the trainer never reaches inside.
- New coordination metrics go into `summarize` so they appear in both train metrics and evals.

## Gotchas

- `pkill -f <pattern>` and `pgrep -f` inside a Bash command match that shell itself and kill it. Use
  `make stop-vllm` / `scripts/stop_vllm.sh`, which selects by `comm=python`.
- The GPU is shared with other experiments (e.g. `/home/rlvr_experiments`). Check `nvidia-smi` before starting
  vLLM or training, and never kill another project's process without asking.
- LFM2 has conv layers, so there is no sequence packing in the trainer (state would leak across rows); it uses SDPA.
- vLLM's LoRA decode for LFM2 is ~15x slower here, so the default is `--sync merge` (merged weights +
  `/collective_rpc reload_weights` + `/reset_prefix_cache`; needs `VLLM_SERVER_DEV_MODE=1`, set in `serve_vllm.sh`).
- `serve_vllm.sh` passes `--generation-config vllm`: LFM's generation config adds repetition penalty and top-k,
  which bias RL sampling.
- LFM2.5 always thinks and emits pythonic tool calls `<|tool_call_start|>[f(a=1)]<|tool_call_end|>`; `proxy.py`
  parses both these and Hermes-style calls.
- If a run crashes mid-way, resume with `--resume-step N` (needs `adapters/step_N`; AdamW state from `optimizer.pt`).

## Conventions

- Match the existing style: module docstring explaining the why, dense code, few comments, 120-column lines,
  no new dependencies without a reason.
- Report results as tables (step × bucket, org vs solo), with paired significance tests where claims depend on it.
  Say what was and was not run.
- Do not commit or push unless asked.
