#!/bin/bash
# Policy server for train.py / probe.py. Shares the 3090 Ti with the trainer.
# Reuses rl_harness's LFM2.5 environment (vLLM 0.32, transformers 5.19, torch 2.13 cu130).
MODEL=${MODEL:-LiquidAI/LFM2.5-2.6B}
VENV=${VENV:-../rl_harness/.venv-lfm}
# Only for train.py --sync lora (Qwen): LORA_ARGS="--enable-lora --max-lora-rank 32 --max-loras 2 --max-cpu-loras 4"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_SERVER_DEV_MODE=1          # /collective_rpc reload_weights + /reset_prefix_cache (train.py --sync merge)
export VLLM_USE_FLASHINFER_SAMPLER=0   # its JIT build needs a newer nvcc than the system one
exec "$(dirname "$0")/$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" --port ${PORT:-8000} \
  --gpu-memory-utilization ${GPU_UTIL:-0.45} --max-model-len ${MAX_LEN:-16384} --max-num-seqs ${MAX_SEQS:-192} \
  ${LORA_ARGS} \
  --return-tokens-as-token-ids --dtype bfloat16 ${EXTRA_ARGS} \
  --generation-config vllm   # ignore LFM generation_config.json (rep penalty 1.1, top_k 50): RL needs plain sampling
