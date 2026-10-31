# multi_agent_rl status

## 2026-10-29: built, tested offline, not yet run on GPU
- Code complete (see README): partitioned-data org environment, lead/worker/solo harnesses, RLCR collective
  advantage, `--train-roles` ablation, tandem-repeat masking, 4 test buckets (seen/tmpl/domain/wide).
- `python org_tasks.py`: shards reassemble exactly (200 train + 160 test tasks checked); train mix easy .2 /
  medium .4 / hard .4, 2–3 sites, ~57% messy.
- `tests/smoke_offline.py` passes: scripted fake vLLM, real proxy + Docker + grading. Orgs with 2/3/4 sites
  fan out in parallel (latency 4), each agent's calls merge into one row, loss tokens == sampled tokens,
  `collective_samples` signs/roles/flat-group handling verified.
- The trainer itself (model on GPU) has not been run here. Its only change from rl_harness is the
  rollout/advantage section, now in `rollout.collective_samples`.
- GPU was busy at build time (an rlvr_experiments countdown_lora run held ~22 GB), so nothing ran on GPU.

## Next
1. `serve_vllm.sh`, then `probe.py --n 10` and `probe.py --show 3`. Check base org vs solo accuracy and
   coverage. If base org accuracy is ~0 (flat groups), lower difficulty (`mix`) or add a worker demo.
   If it is ~90%+, raise the hard share or add 4-site tasks to training.
2. `train.py --run org_all` (60 steps), then the `--train-roles lead` ablation.
3. Ideas: peer channels between workers (Avalon message board), lookup tables held at only one site
   (forces lead-mediated joins), a competitive or mixed-motive environment with SPIRAL-style
   role-conditioned advantages.

## Gotchas (inherited from rl_harness)
- `pkill -f <pattern>` / `pgrep -f` inside a Bash command also matches that shell. Find vLLM via
  `ps -eo pid,comm,args | grep vllm` and kill by pid.
- LFM2: no sequence packing (conv state leaks across packed rows), SDPA, merged-weight sync (vLLM LoRA decode
  for LFM2 is ~15x slower on this box), `--generation-config vllm`.
