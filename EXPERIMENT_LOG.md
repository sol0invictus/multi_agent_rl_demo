# Multi-Agent RL Experiment Log (`runs/org_small` & `runs/org_small_lead`)

**Dates**: 2026-10-31 (`org_small`) / 2026-11-01 (`org_small_lead`)  
**Branch**: `dev` (merged into `main`)  
**Run Directories**: `runs/org_small/` (`--train-roles all`) and `runs/org_small_lead/` (`--train-roles lead`)  
**Log Files**: `runs/org_small.log`, `runs/org_small_lead.log`

---

## 1. Run Configuration & Setup

| Parameter | Value | Notes |
|---|---|---|
| `--run` | `org_small` | Saved under `runs/org_small/` |
| `--model` | `LiquidAI/LFM2.5-2.6B` | Shared policy across `lead`, `worker`, and `solo` |
| `--mode` | `org` | Trained **only** on multi-agent organizations (`lead` + `K` per-site `worker`s) |
| `--eval-modes` | `org,solo` | Evaluated on both `org` and single-agent `solo` |
| `--train-roles` | `all` | Avalon-M1 RLCR: every sampled token of every agent gets the collective advantage |
| `--rank` | `32` (`lora_alpha=32`) | Attention (`q/k/v/out_proj`) + MLP (`w1/w2/w3`); `40,271,872` / `2,737,470,464` params (`1.4711%`) |
| `--lr` | `4e-5` | AdamW (`betas=(0.9, 0.99)`, `weight_decay=0.0`, grad clip `1.0`) |
| `--steps` | `30` | Steps `1–10` (`--eval-every 5`), resumed `11–30` (`--resume-step 10 --eval-every 10`) |
| `--tasks-per-step` | `4` | `4` tasks × `group 6` = `24` organization rollouts per step (`120` tasks total across 30 steps) |
| `--group` | `6` | GRPO group size per task |
| `--n-test` | `10` | `10` tasks × `4` buckets (`seen`, `tmpl`, `domain`, `wide`) = `40` test tasks per mode (`80` total eval rollouts) |
| `--eff-bonus` | `0.1` | Up to `+0.1` reward bonus for short critical-path latency |
| `--is-cap` | `2.0` | Truncated importance sampling cap against vLLM logprobs |
| `--adv-std-floor` | `0.05` | Group standard-deviation floor |
| `--temperature` / `--eval-temperature` | `1.0` / `0.7` | Rollout vs evaluation sampling temperature |
| `--sync` | `merge` | Full merged weights (`model.safetensors`) reloaded into vLLM via `/collective_rpc` + `/reset_prefix_cache` |

### Commands Executed
```bash
# 1. Start vLLM policy server (~45% of RTX 3090 Ti VRAM)
mkdir -p runs && ./serve_vllm.sh > runs/vllm.log 2>&1 &

# 2. Initial 10-step run (eval at steps 0, 5, 10)
../rl_harness/.venv-lfm/bin/python -u train.py \
  --run org_small --steps 10 --tasks-per-step 4 --group 6 --n-test 10 --eval-every 5 \
  2>&1 | tee runs/org_small.log

# 3. Resumed from step 10 to step 30 (eval at steps 20, 30)
../rl_harness/.venv-lfm/bin/python -u train.py \
  --run org_small --resume-step 10 --steps 30 --tasks-per-step 4 --group 6 --n-test 10 --eval-every 10 \
  2>&1 | tee -a runs/org_small.log
```

---

## 2. Evaluation Results (Steps 0, 5, 10, 20, 30)

### 2.1 Accuracy by Bucket & Mode (`n = 10` per bucket, `40` total per mode)

| Step | `org` All | `org` seen | `org` tmpl | `org` domain | `org` wide ($K=4$) | `solo` All | `solo` seen | `solo` tmpl | `solo` domain | `solo` wide ($K=4$) |
|---|---|---|---|---|---|---|---|---|---|---|
| **0 (Base)** | **0.225** (`9/40`) | 0.200 | 0.400 | 0.200 | 0.100 | **0.450** (`18/40`) | 0.500 | 0.400 | 0.400 | 0.500 |
| **5** | **0.250** (`10/40`) | 0.300 | 0.200 | 0.500 | 0.000 | **0.400** (`16/40`) | 0.500 | 0.600 | 0.200 | 0.300 |
| **10** | **0.300** (`12/40`) | 0.200 | 0.400 | 0.500 | 0.100 | **0.450** (`18/40`) | 0.600 | 0.500 | 0.400 | 0.300 |
| **20** | **0.375** (`15/40`) | 0.400 | 0.600 | 0.400 | 0.100 | **0.425** (`17/40`) | 0.200 | 0.400 | 0.500 | 0.600 |
| **30** | **0.375** (`15/40`) | 0.300 | **0.600** | 0.400 | **0.200** | **0.625** (`25/40`) | **0.700** | **0.700** | **0.700** | 0.400 |

### 2.2 Accuracy by Site Count ($K$)

| Step | `org` $K=2$ (`n=12`) | `org` $K=3$ (`n=18`) | `org` $K=4$ (`n=10`) | `solo` $K=2$ (`n=12`) | `solo` $K=3$ (`n=18`) | `solo` $K=4$ (`n=10`) |
|---|---|---|---|---|---|---|
| **0** | 0.167 | 0.333 | 0.100 | 0.250 | 0.556 | 0.500 |
| **5** | 0.167 | 0.444 | 0.000 | 0.500 | 0.389 | 0.300 |
| **10** | 0.250 | 0.444 | 0.100 | 0.583 | 0.444 | 0.300 |
| **20** | **0.417** | **0.500** | 0.100 | 0.167 | 0.500 | 0.600 |
| **30** | **0.417** | 0.444 | **0.200** | **0.750** | **0.667** | 0.400 |

### 2.3 Coordination & Efficiency Metrics in `org` Mode

| Step | Cov / Full Cov | Acc│Full Cov | Fanout | Latency | Lead Turns / Asks | Worker Calls / Tool Calls | Reply Fallbacks | Worker Exhausted | Status (`submitted` / `final_text` / `max_turns`) | Gen Tokens (`lead` / `worker` / `solo`) | Eval Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 1.00 / 1.00 | 0.225 | 2.15 | 18.3 | 4.18 / 4.80 | 19.8 / 18.9 | 3.20 | 1.18 | `7` / `27` / `6` | 2,829 / 9,320 / 2,787 | 469s |
| **5** | 0.98 / 0.95 | 0.263 | 2.15 | 16.3 | 4.00 / 4.53 | 17.8 / 16.4 | 3.28 | 0.73 | `7` / `28` / `5` | 2,612 / 8,584 / 2,703 | 407s |
| **10** | 0.99 / 0.98 | 0.308 | 1.95 | 17.1 | 4.10 / 4.30 | 16.9 / 14.9 | 2.75 | 0.60 | `6` / `29` / `5` | 3,152 / 8,094 / 2,280 | 420s |
| **20** | 1.00 / 1.00 | 0.375 | 2.05 | 15.9 | 4.00 / 4.38 | 16.9 / 15.7 | 2.63 | 0.73 | `10` / `27` / `3` | 2,456 / 8,102 / 3,643 | 406s |
| **30** | **1.00 / 1.00** | **0.375** | 2.00 | **14.1** | **3.63 / 3.75** | **14.4 / 12.8** | **2.35** | **0.53** | `7` / `32` / **`1`** | **2,374 / 7,224 / 2,397** | **332s** |

### 2.4 Paired Significance Tests (Exact Two-Sided McNemar vs Step 0)

| Comparison | Mode | Step 0 | Step $S$ | Fixed ($0 \to S$) | Broke ($0 \to S$) | Exact McNemar $p$ |
|---|---|---|---|---|---|---|
| **Step 0 → 10** | `org` (`n=40`) | 9/40 (0.225) | 12/40 (0.300) | 8 | 5 | 0.581 |
| **Step 0 → 10** | `solo` (`n=40`) | 18/40 (0.450) | 18/40 (0.450) | 7 | 7 | 1.000 |
| **Step 0 → 10** | `all` (`n=80`) | 27/80 (0.338) | 30/80 (0.375) | 15 | 12 | 0.701 |
| **Step 0 → 20** | `org` (`n=40`) | 9/40 (0.225) | 15/40 (0.375) | 8 | 2 | 0.109 |
| **Step 0 → 20** | `solo` (`n=40`) | 18/40 (0.450) | 17/40 (0.425) | 5 | 6 | 1.000 |
| **Step 0 → 20** | `all` (`n=80`) | 27/80 (0.338) | 32/80 (0.400) | 13 | 8 | 0.383 |
| **Step 0 → 30** | `org` (`n=40`) | 9/40 (0.225) | **15/40 (0.375)** | **8** | **2** | 0.109 |
| **Step 0 → 30** | `solo` (`n=40`) | 18/40 (0.450) | **25/40 (0.625)** | **13** | **6** | 0.167 |
| **Step 0 → 30** | **`all` (`n=80`)** | **27/80 (0.338)** | **40/80 (0.500)** | **21** | **8** | **0.0241** |

---

## 3. Training Curve (`runs/org_small/metrics.jsonl`)

### 3.1 Five-Step Averages

| Steps | Reward | Acc | Latency | Coverage | Max Fanout | Asks | Flat Groups | Errors | Mismatch KL | Grad Norm | Rollout Time | Train Time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1–5** | 0.208 | 0.200 | 16.40 | 0.98 | 1.84 | 4.33 | 1.60 / 4 | 0.00 | 0.0004 | 0.035 | 256.89s | 181.26s |
| **6–10** | 0.298 | 0.283 | 15.50 | 0.99 | 1.78 | 3.98 | 2.40 / 4 | 0.00 | 0.0003 | 0.056 | 276.62s | 136.43s |
| **11–15** | 0.251 | 0.242 | 14.66 | 0.98 | 1.76 | 3.83 | 1.20 / 4 | 0.00 | 0.0004 | 0.038 | 254.22s | 198.00s |
| **16–20** | 0.280 | 0.267 | 14.38 | 0.96 | 1.88 | 3.70 | 1.60 / 4 | 0.00 | 0.0004 | 0.040 | 245.87s | 189.74s |
| **21–25** | 0.296 | 0.283 | 14.89 | 0.99 | 1.88 | 3.87 | 1.60 / 4 | 0.00 | 0.0003 | 0.027 | 242.57s | 170.06s |
| **26–29** | **0.371** | **0.354** | **13.00** | **0.99** | 1.80 | **3.38** | **0.75 / 4** | 0.00 | 0.0004 | 0.035 | **222.42s** | 186.65s |

### 3.2 Per-Step Training Log

| Step | Reward | Acc | Lat | Cov | Fan | Asks | Subm | W.Exh | Flat | Rows | Loss Tok (`lead` / `worker`) | KL | Grad Norm | Roll / Train (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.132 | 0.125 | 14.5 | 0.96 | 1.54 | 3.9 | 0.167 | 0.38 | 1/4 | 58 | 73,269 / 96,510 | 0.0003 | 0.029 | 197 / 210 |
| 2 | 0.045 | 0.042 | 19.1 | 0.96 | 1.75 | 5.0 | 0.042 | 1.25 | 3/4 | 19 | 17,546 / 72,818 | 0.0006 | 0.029 | 339 / 73 |
| 3 | 0.300 | 0.292 | 19.8 | 1.00 | 2.17 | 5.2 | 0.125 | 1.00 | 1/4 | 68 | 47,504 / 175,638 | 0.0005 | 0.027 | 321 / 246 |
| 4 | 0.389 | 0.375 | 16.1 | 1.00 | 1.96 | 4.4 | 0.167 | 0.58 | 1/4 | 67 | 32,643 / 120,635 | 0.0003 | 0.030 | 226 / 219 |
| 5 | 0.173 | 0.167 | 12.5 | 0.98 | 1.79 | 3.2 | 0.208 | 0.50 | 2/4 | 42 | 16,299 / 53,838 | 0.0002 | 0.058 | 202 / 158 |
| 6 | 0.178 | 0.167 | 19.3 | 1.00 | 1.75 | 5.1 | 0.083 | 1.08 | 3/4 | 18 | 4,995 / 25,892 | 0.0002 | 0.066 | 347 / 135 |
| 7 | 0.390 | 0.375 | 14.6 | 0.98 | 2.17 | 3.9 | 0.125 | 0.96 | 2/4 | 48 | 18,874 / 93,487 | 0.0005 | 0.042 | 234 / 180 |
| 8 | 0.222 | 0.208 | 17.1 | 1.00 | 1.62 | 4.1 | 0.042 | 0.75 | 3/4 | 18 | 9,679 / 23,742 | 0.0002 | 0.091 | 294 / 45 |
| 9 | 0.215 | 0.208 | 13.8 | 0.96 | 1.62 | 3.7 | 0.250 | 0.29 | 2/4 | 36 | 28,733 / 66,762 | 0.0004 | 0.043 | 196 / 168 |
| 10 | 0.484 | 0.458 | 12.7 | 1.00 | 1.75 | 3.2 | 0.000 | 0.50 | 2/4 | 36 | 6,866 / 58,441 | 0.0004 | 0.041 | 312 / 156 |
| 11 | 0.214 | 0.208 | 16.9 | 1.00 | 1.79 | 4.4 | 0.167 | 0.79 | 1/4 | 61 | 63,787 / 187,793 | 0.0004 | 0.030 | 277 / 268 |
| 12 | 0.133 | 0.125 | 14.9 | 0.99 | 2.29 | 4.8 | 0.125 | 0.67 | 2/4 | 42 | 34,411 / 76,051 | 0.0004 | 0.038 | 324 / 90 |
| 13 | 0.306 | 0.292 | 14.8 | 1.00 | 1.50 | 3.5 | 0.458 | 0.46 | 2/4 | 37 | 26,260 / 57,175 | 0.0004 | 0.049 | 249 / 161 |
| 14 | 0.215 | 0.208 | 11.9 | 0.90 | 1.46 | 2.8 | 0.375 | 0.54 | 1/4 | 50 | 42,206 / 115,670 | 0.0004 | 0.041 | 224 / 207 |
| 15 | 0.388 | 0.375 | 14.8 | 1.00 | 1.75 | 3.6 | 0.167 | 0.46 | 0/4 | 84 | 65,928 / 173,504 | 0.0004 | 0.030 | 197 / 264 |
| 16 | 0.349 | 0.333 | 13.5 | 0.96 | 1.79 | 3.5 | 0.125 | 0.54 | 0/4 | 83 | 47,264 / 196,093 | 0.0004 | 0.031 | 230 / 265 |
| 17 | 0.090 | 0.083 | 15.1 | 0.90 | 1.79 | 3.8 | 0.167 | 0.67 | 2/4 | 40 | 26,563 / 74,230 | 0.0005 | 0.034 | 235 / 183 |
| 18 | 0.133 | 0.125 | 14.3 | 0.94 | 2.04 | 3.9 | 0.250 | 0.46 | 3/4 | 23 | 17,916 / 68,605 | 0.0003 | 0.060 | 300 / 161 |
| 19 | 0.302 | 0.292 | 15.1 | 0.97 | 2.00 | 3.9 | 0.083 | 0.50 | 2/4 | 46 | 21,945 / 53,273 | 0.0004 | 0.032 | 203 / 158 |
| 20 | 0.527 | 0.500 | 13.9 | 1.00 | 1.79 | 3.5 | 0.125 | 0.71 | 1/4 | 60 | 16,668 / 89,938 | 0.0004 | 0.041 | 261 / 182 |
| 21 | 0.304 | 0.292 | 16.0 | 0.99 | 2.08 | 4.1 | 0.208 | 0.63 | 1/4 | 73 | 45,392 / 145,879 | 0.0004 | 0.031 | 275 / 229 |
| 22 | 0.127 | 0.125 | 17.2 | 1.00 | 1.62 | 4.0 | 0.125 | 0.75 | 2/4 | 42 | 56,574 / 121,168 | 0.0004 | 0.029 | 257 / 201 |
| 23 | 0.000 | 0.000 | 16.2 | 1.00 | 2.17 | 4.5 | 0.208 | 0.67 | 4/4 | 0 | 0 / 0 | 0.0000 | 0.000 | 275 / 15 |
| 24 | 0.262 | 0.250 | 14.1 | 1.00 | 1.71 | 3.8 | 0.208 | 0.50 | 1/4 | 55 | 29,488 / 123,200 | 0.0004 | 0.037 | 270 / 205 |
| 25 | 0.787 | 0.750 | 11.0 | 0.96 | 1.79 | 3.0 | 0.125 | 0.21 | 0/4 | 82 | 26,390 / 101,852 | 0.0004 | 0.036 | 136 / 200 |
| 26 | 0.488 | 0.458 | 12.8 | 1.00 | 1.79 | 3.1 | 0.125 | 0.46 | 0/4 | 78 | 51,784 / 172,707 | 0.0004 | 0.026 | 251 / 244 |
| 27 | 0.301 | 0.292 | 14.0 | 1.00 | 2.17 | 4.3 | 0.167 | 0.63 | 1/4 | 73 | 44,507 / 136,419 | 0.0004 | 0.032 | 238 / 221 |
| 28 | 0.522 | 0.500 | 12.5 | 1.00 | 1.54 | 3.1 | 0.042 | 0.46 | 1/4 | 54 | 25,606 / 100,664 | 0.0004 | 0.038 | 194 / 182 |
| 29 | 0.171 | 0.167 | 12.7 | 0.96 | 1.71 | 3.0 | 0.333 | 0.46 | 1/4 | 52 | 47,204 / 100,260 | 0.0006 | 0.045 | 207 / 99 |

*(Note: Step 30 completed its gradient step, saved `adapters/step_30`, `optimizer.pt`, and `policy/model.safetensors`, and synced merged weights to vLLM; see Section 5 for the logging bug fix applied after Step 30.)*

---

## 4. Key Findings & Task-Level Analysis (Step 0 → Step 30)

1. **Multi-Agent (`org`) Accuracy (+15.0 pp, `22.5% → 37.5%`)**:
   - **8 tasks fixed, only 2 broken**:
     - Fixed: `o19900001` (`seen/hr/avg_salary_dept`, $K=2$, `90,095.65 → $85,105.62` via weighted sum/count vs unweighted average of site averages), `o19900007` (`seen/retail/top_product_qty`, $K=2$, latency `12 → 5`), `o20000000` (`tmpl/hr/min_rating_year`, $K=3$), `o20000009` (`tmpl/hr/min_rating_year`, $K=2$), `o20100003` (`domain/web/pct_device`, $K=3$, `49.0% → 48.0%`, latency `24 → 15`), `o20100006` (`domain/web/pct_device`, $K=3$, `max_turns (36) → 38.8% (11)`), `o20200002` (`wide/retail/busiest_month`, $K=4$, `max_turns (33) → 7 (8)`), `o20200003` (`wide/hr/n_hired_after`, $K=4$, `154 → 156`).
     - Broken: `o19900003` (`seen/hr/top_hire_year_dept`, $K=3$, lead stopped via `final_text` mid-summary) and `o20200009` (`wide/retail/rev_region`, $K=4$, `59110.46 → 59299.03` duplicate-row filtering miss at one site).
2. **Large Efficiency & Coordination Gains in `org` Mode**:
   - Critical-path latency fell **22.8%** (`18.3 → 14.1` model calls).
   - Worker turn exhaustion fell **55.3%** (`1.18 → 0.53` per task), cutting organization `max_turns` timeouts from **`6/40` to `1/40`**.
   - Worker plain-text `reply_fallbacks` fell **26.6%** (`3.20 → 2.35`).
3. **Zero-Shot Transfer from `org` Training to `solo` Evaluation (+17.5 pp, `45.0% → 62.5%`)**:
   - Even though `solo` was never trained, shared-weight RLCR training on `org` fixed **13 `solo` tasks** (including all 5 Step-0 `max_turns` timeouts and all 5 Step-0 malformed `<|tool_call_start|>` syntax errors) while breaking 6.
   - Combined `org + solo` accuracy rose from **`27/80 (33.8%)` to `40/80 (50.0%)`** (exact two-sided McNemar **$p = 0.0241$**).
4. **Main Remaining Bottleneck (`final_text` in `org_harness`)**:
   - In `agents.py`, if the lead outputs a message with no tool calls, `org_harness` immediately terminates with `status = "final_text"`. At Step 30, `32/40` eval org rollouts ended with `final_text` (`7/40` with `submitted`, `1/40` with `max_turns`). When the lead states the final number at the end of its plain-text message, `tasks.grade` still grades it as correct, but when the lead writes an intermediate summary (*"Now I have data from all three sites..."*) without calling `submit`, the rollout ends prematurely.

---

## 5. Code Fixes & Saved Checkpoints

### Bug Fix Applied (`rollout.py`)
- In `rollout.run_rollout` (`rollout.py` line 41), added `"n_sites": task.n_sites` to the `except Exception` fallback dictionary so that any rollout raising an exception does not trigger `KeyError: 'n_sites'` when computing per-$K$ accuracy in `train.py` or `rollout.evaluate`.

### Saved Artifacts on Disk
- **LoRA Adapters**: `runs/org_small/adapters/step_1` through `runs/org_small/adapters/step_30`
- **Optimizer State**: `runs/org_small/optimizer.pt` (AdamW state at step 30)
- **Merged Policy Weights**: `runs/org_small/policy/model.safetensors` (step 30 merged bfloat16 weights)
- **Metrics & Rollout Logs**:
  - `runs/org_small/config.json`
  - `runs/org_small/metrics.jsonl` (steps 1–29)
  - `runs/org_small/eval.jsonl` (steps 0, 5, 10, 20, 30)
  - `runs/org_small/eval_rollouts.jsonl` (full per-task eval results for steps 0, 5, 10, 20, 30)
  - `runs/org_small/train_rollouts.jsonl` (training rollout records for steps 1–29)

---

## 6. How to Resume or Run Next Experiments

### Option A: Continue `org_small` from Step 30 (e.g. to Step 50)
```bash
# If vLLM is not running, start it first:
make serve

# Resume training from step 30:
../rl_harness/.venv-lfm/bin/python -u train.py \
  --run org_small --resume-step 30 --steps 50 --tasks-per-step 4 --group 6 --n-test 10 --eval-every 10 \
  2>&1 | tee -a runs/org_small.log
```

### Option B: Completed `--train-roles lead` Ablation (`runs/org_small_lead`)
```bash
make stop-vllm
make serve   # wait ~25s for :8000/health

../rl_harness/.venv-lfm/bin/python -u train.py \
  --run org_small_lead --train-roles lead --steps 30 --tasks-per-step 4 --group 6 --n-test 10 --eval-every 10 \
  2>&1 | tee runs/org_small_lead.log

# Compare org_small vs org_small_lead with paired McNemar test:
../rl_harness/.venv-lfm/bin/python analysis/summarize_run.py org_small --compare org_small_lead
```

| Step | `org` All | `org` seen | `org` tmpl | `org` domain | `org` wide ($K=4$) | `solo` All | `solo` seen | `solo` tmpl | `solo` domain | `solo` wide ($K=4$) | `org` Latency | Worker Exhausted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0 (Base)** | 0.300 (`12/40`) | 0.300 | 0.500 | 0.200 | 0.200 | 0.425 (`17/40`) | 0.300 | 0.300 | 0.400 | 0.700 | 18.7 | 1.05 |
| **10** | 0.125 (`5/40`) | 0.000 | 0.200 | 0.200 | 0.100 | 0.375 (`15/40`) | 0.000 | 0.700 | 0.300 | 0.500 | 15.8 | 1.00 |
| **20** | 0.275 (`11/40`) | 0.200 | 0.300 | 0.500 | 0.100 | 0.425 (`17/40`) | 0.600 | 0.500 | 0.200 | 0.400 | 16.7 | 1.05 |
| **30** | **0.275 (`11/40`)** | 0.400 | 0.400 | 0.300 | **0.000** | **0.500 (`20/40`)** | 0.500 | 0.300 | 0.700 | 0.500 | **16.8** | **1.10** |

### Option C: Full-Scale 60-Step Run (`org_all`, 8 tasks × group 8, 30 test tasks/bucket)
```bash
make stop-vllm
make serve
make train RUN=org_all ARGS="--steps 60 --eval-every 20"
```
