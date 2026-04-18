# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

OpenClaw-RL is not a single codebase — it is a collection of **self-contained method folders** that all plug into the shared `slime/` training framework and `Megatron-LM/`. Each top-level folder is an independent RL method or environment:

- **Track 1 — Personal Agent Optimization** (online RL from conversation feedback; all talk to OpenClaw via an OpenAI-compatible proxy on port `30000`):
  - `openclaw-rl/` — Binary RL (GRPO) with PRM scoring
  - `openclaw-opd/` — On-Policy Distillation with hindsight hints + teacher log-probs
  - `openclaw-combine/` — Weighted combination of Binary RL + OPD (recommended)
  - `openclaw-tinker/` — Same three methods, but running on [Tinker](https://tinker.build) cloud (LoRA only, no local GPUs)
  - `openclaw-test/` — End-to-end evaluation harness (GSM8K student/teacher role-play)
- **Track 2 — Agentic RL in real-world settings** (each has its own environment server and rollout loop):
  - `terminal-rl/` — Terminal agent, pool of remote Docker workers (`remote/` subdir)
  - `gui-rl/` — VLM GUI agent, cloud VM pool (Volcengine/AWS/Aliyun)
  - `swe-rl/` — SWE-Bench agent, remote ECS Docker nodes (`swe_exec_server.py` on :5000, pool server on :18090)
  - `toolcall-rl/` — Math + Python interpreter, local sandbox
- **Shared infrastructure:**
  - `slime/` — RL framework (Megatron + SGLang + router + data buffer); see `slime/slime/`
  - `Megatron-LM/` — Training backend (vendored); `train_rl.py` is the RL entry point
  - `instructions/README.md` — Canonical env-setup recipe (CUDA 12.9, Python 3.12, DeepEP, apex, flash-attn, flashinfer, megatron-bridge, TransformerEngine)
  - `extensions/rl-training-headers/` — TypeScript OpenClaw plugin for routing requests to an RL server

### Hard rule: don't modify shared framework code

**Do not edit `slime/`, `Megatron-LM/`, or `openclaw/` unless absolutely necessary.** The framework exposes extension points so new methods can plug in without touching shared code — use them. If a framework change is truly required, call it out explicitly and keep it in its own commit/PR with a clear justification.

When adding a new method, create a new top-level folder parallel to `openclaw-rl/`. When extending an existing method (new model family, LoRA variant, low-precision example), **add new files rather than modifying existing ones** so the original working examples stay intact.

## Common commands

### Launching training

Every Track 1 and Track 2 training script is launched from inside `slime/` so it picks up slime's model configs and `train_async.py`:

```bash
cd slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine.sh   # personal agent, recommended
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh             # binary RL
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh           # OPD
bash ../terminal-rl/terminal_qwen3_8b_rl.sh                 # terminal agent
bash ../gui-rl/gui_qwen3vl_8b_rl.sh                         # GUI agent
bash ../swe-rl/scripts/run_swe_rl_32b_remote_8nodes.sh      # SWE agent
bash ../toolcall-rl/retool_qwen3_4b_rl.sh                   # tool-call agent
```

LoRA variants exist as sibling scripts (`*_lora.sh`). Qwen3.5 variants are named `run_qwen35_*`. Tinker runs use `cd openclaw-tinker && python run.py --method {rl,opd,combine} ...`.

Each script starts a local Ray head (`ray start --head`), then `ray job submit` runs `train_async.py` with a long arg list. At the top of every script there is a brutal `pkill -9 sglang / ray stop --force / pkill -9 python / pkill -9 ray` cleanup — expect training to kill any existing Ray/SGLang on the node.

### Inputs via environment variables

Launch scripts read these; prefer exporting before calling rather than editing the script:

| Variable | Meaning |
|---|---|
| `HF_CKPT` | HuggingFace checkpoint for the policy model |
| `REF_LOAD` | Reference model for KL regularization (often `$HF_CKPT`, but resume scripts need pre-converted `torch_dist`) |
| `SAVE_CKPT` | Output checkpoint dir |
| `PRM_MODEL_PATH` | Process Reward Model / judge checkpoint |
| `PROMPT_DATA` / `ROLLOUT_PROMPT_DATA` | Training JSONL |
| `NUM_GPUS`, `ACTOR_GPUS`, `ROLLOUT_GPUS`, `PRM_GPUS` | GPU partition; `ACTOR+ROLLOUT+PRM ≤ NUM_GPUS` is enforced |
| `SGLANG_API_KEY` | Auth token for the served policy API; must match OpenClaw client |
| `PORT` | Policy API port (default `30000`) |
| `PRM_M` | PRM / hint-judge majority-vote count |
| `OPENCLAW_COMBINE_W_RL`, `OPENCLAW_COMBINE_W_OPD` | Advantage weights (combine method only) |
| `TRAIN_EPOCHS` | Sample-duplication factor per rollout batch (combine typically `2`) |
| `OPENCLAW_RECORD_ENABLED`, `OPENCLAW_RECORD_FILE` | Dump rollout records to JSONL |
| `OPENCLAW_EVAL_MODE` | Enable W&B eval logging (defaults on for combine/opd) |
| `WANDB_KEY`, `WANDB_PROJECT` | W&B logging |
| `MASTER_ADDR` | Ray head node IP for multi-node runs |

Track 2 environments add their own: `WORKER_URLS` (terminal), `SWE_EXEC_SERVER_URLS` (SWE), `GUI_PROVIDER_NAME` + `VOLC_ACCESSKEY`/`VOLC_SECRETKEY` (GUI).

### Environment setup

Follow `instructions/README.md` verbatim — it is pinned to specific versions (torch 2.9.1+cu129, DeepEP, apex, flash-attn 2.7.4.post1, flashinfer-jit-cache 0.6.3, megatron-bridge, transformer_engine 2.10.0). Qwen3.5 requires `transformers==5.3.0` and, for multimodal, a separate `Megatron-Bridge-qwen35` checkout at a pinned SHA. Do not `pip install -U` anything without checking.

### Tests / checks

There is no project-wide test runner. The only tests live under `slime/tests/` and `Megatron-LM/tests/` and are unit/integration tests for the framework itself; they are not wired to the method folders here. For method changes, the smoke test is actually launching the relevant `run_*.sh` and checking the Ray dashboard + W&B.

## Architecture: fully-async 4-component loop

All methods in this repo share the same asynchronous decoupling — this is the load-bearing idea of the codebase.

```
         OpenClaw / env                         slime (Ray)
┌──────────────────────────┐        ┌────────────────────────────────┐
│ user / environment turns │        │  Megatron actor (train)        │
└───────────┬──────────────┘        │  SGLang rollout engine         │
            │ OpenAI-compatible      │  PRM/judge engine (optional)   │
            ▼                        └──────────▲─────────────────────┘
┌──────────────────────────┐                    │ samples
│  openclaw_*_api_server   │  ◄── forwards ──►  │
│  (FastAPI proxy)         │                    │
│  PRM eval + hint judge   │                    │
│  sample submission       │────────────────────┘
└──────────────────────────┘
```

The **method folder owns** the proxy + rollout + loss, wired into slime via these flags on `train_async.py`:

- `--custom-generate-function-path <module>.generate` — method's FastAPI/proxy entry
- `--custom-rm-path <module>.reward_func` — how the trainer pulls ready samples (blocking)
- `--rollout-function-path <module>.generate_rollout_*` — async rollout worker
- `--custom-loss-function-path <module>.<fn>` + `--loss-type custom_loss` — custom advantage/loss (OPD top-K, combine, etc.)
- `--disable-rollout-global-dataset` — we stream samples from live conversations, not a pre-collected dataset
- `--disable-rewards-normalization` — binary rewards should not be z-scored
- `--advantage-estimator grpo` with `--n-samples-per-prompt 1` is the default for online streaming

`PYTHONPATH` is injected via the Ray `runtime_env_json` and points at `Megatron-LM/`, `slime/`, and the method folder (sometimes also `openclaw-opd/` when combine needs to reuse its hint-judge code).

### Samples, scoring, and the "at-least-one guarantee"

- Conversations are split into **main-line** (trainable) and **side** (non-trainable) turns.
- For each main-line turn, when the **next turn** arrives, its content is the "next state" used for scoring.
- A turn that never receives a next state (last turn in a session) is excluded (`loss_mask=0`), **unless** it is the only turn — the "at-least-one guarantee" forces the best single turn to get `reward=+1` so no session is wasted.
- PRM/hint-judge runs `--prm-m` votes concurrently and takes a majority; votes produce `+1 / -1 / 0`.
- Combine's three-way dispatch: a single turn can emit an OPD sample, an RL sample, or both; `reward=0` samples get zero GRPO advantage and `teacher_logp ≈ rollout_logp` samples get zero teacher advantage, so the unified loss naturally routes each sample to its correct branch.

### Launch script layout convention

Every launch script groups args into named bash arrays (`CKPT_ARGS`, `ROLLOUT_ARGS`, `PERF_ARGS`, `GRPO_ARGS`/`COMBINE_ARGS`, `OPTIMIZER_ARGS`, `SGLANG_ARGS`, `PRM_ARGS`, `CUSTOM_ARGS`, `MISC_ARGS`, `WANDB_ARGS`) and unions them at the `ray job submit` call. Keep this structure when adding new scripts.

### Remote workers (Track 2)

- **terminal-rl:** remote pool servers on each worker machine at `:18081`; set `WORKER_URLS="http://w1:18081,..."` on the training node. See `terminal-rl/remote/README.md`.
- **swe-rl:** each Docker node runs `swe_exec_server.py` on `:5000` as a systemd service (set up once via `server/setup_ecs_seed.sh`, which takes 2–5 hours to pull SWE-Bench images — snapshot the VM afterwards); head node runs `swe_env_pool_server.py` on `:18090`; set `SWE_EXEC_SERVER_URLS` to the comma-separated node list. The `_resume.sh` 32B script requires a pre-converted Megatron `torch_dist` checkpoint (one-time run of `slime/tools/convert_hf_to_torch_dist.py`). Everything else loads from HF via bridge mode.
- **gui-rl:** Flask env pool server manages a pool of cloud VMs, pre-warming `GUI_PREWARM_ENVS` instances before training starts to avoid cold-start latency; Volcengine secrets must be exported.

## Key environment vars & secrets

Never commit these. `SGLANG_API_KEY`, `OPENCLAW_GATEWAY_TOKEN`, `WANDB_KEY`/`WANDB_API_KEY`, `VOLC_ACCESSKEY`/`VOLC_SECRETKEY`, `TINKER_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN` — all are read from the shell and should stay in the shell.
