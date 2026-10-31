# Common tasks. Python comes from rl_harness's LFM venv; see CLAUDE.md.
PY    ?= ../rl_harness/.venv-lfm/bin/python
RUN   ?= org_all
ARGS  ?=
N     ?= 10

.PHONY: test tasks smoke image serve stop-vllm probe show train ablation table clean

test: tasks smoke                      ## no-GPU checks

tasks:                                 ## shards reassemble, held-out splits are disjoint
	$(PY) org_tasks.py

smoke:                                 ## end-to-end with a scripted fake vLLM + real Docker sandbox
	$(PY) tests/smoke_offline.py

image:                                 ## build the sandbox image (python + pandas, no network)
	docker build -t harness-sandbox:latest sandbox_image

serve:                                 ## vLLM policy server on :8000 (background)
	mkdir -p runs && nohup ./serve_vllm.sh > runs/vllm.log 2>&1 & echo "vLLM starting, log: runs/vllm.log"

stop-vllm:
	./scripts/stop_vllm.sh

probe:                                 ## base-model org vs solo, N tasks per bucket
	mkdir -p runs && $(PY) probe.py --n $(N) --out runs/probe_base.jsonl $(ARGS)

show:                                  ## print 3 full organisation transcripts
	$(PY) probe.py --show 3 $(ARGS)

train:                                 ## RLCR training run in the background
	mkdir -p runs && nohup $(PY) train.py --run $(RUN) $(ARGS) > runs/$(RUN).log 2>&1 & echo "log: runs/$(RUN).log"

ablation:                              ## same setup, only the lead's tokens are trained
	$(MAKE) train RUN=$(RUN)_lead ARGS="--train-roles lead $(ARGS)"

table:                                 ## markdown results for runs/$(RUN)
	$(PY) analysis/summarize_run.py $(RUN) $(ARGS)

clean:                                 ## leftover sandbox workspaces and caches (keeps runs)
	rm -rf runs/workspaces __pycache__ */__pycache__
