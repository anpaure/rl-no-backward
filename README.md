# RL Without Backward Passes

Can a small language model learn from reinforcement learning without executing
a backward pass? This repository compares ordinary GRPO against a strict
inference-only policy optimizer on GSM8K.

## Head-to-head contract

The corrected comparison uses the same policy and evidence on both sides:

- **Model:** pinned `Qwen/Qwen2.5-1.5B-Instruct`, with the BF16 base frozen.
- **Policy parameters:** standard PEFT LoRA on `q_proj` and `v_proj` in all 28
  transformer blocks, rank 8 and alpha 16: exactly **1,089,536** FP32 trainable
  scalars.
- **Task:** full-difficulty GSM8K with deterministic exact numeric-match reward.
- **Rollouts:** vLLM 0.22, FlashAttention-2, one shared prompt schedule, seeds,
  temperatures, group size, response budget, and versioned LoRA state.
- **Evaluation:** one committed 256-example development partition selected from
  previously untouched official-test IDs. A separate 679-example final
  partition remains locked until methods and hyperparameters are frozen.
- **BP-GRPO:** reverse-mode AdamW on the standard clipped, token-local GRPO
  objective. The fixed-rollout loss and gradient are differentially checked
  against pinned upstream TRL 1.10.
- **FO-NPG:** the same LoRA tensors and fixed-rollout objective, but policy
  derivatives and Fisher coordinates come from symmetric teacher-forced
  inference at `a + mu*v` and `a - mu*v`; a KL-constrained line search applies
  the update. No reverse-mode operation is allowed in its training process.

The first required result is a short integration and learning gate. The longer
multi-seed sweep is launched only after normal GRPO demonstrably changes the
policy and improves the predeclared learning metric, the forward-only
finite-difference coordinates agree with an exact diagnostic gradient, and
both methods reproduce the same initial rollout.

## Why rescore stored trajectories?

Return-difference evolution strategies must regenerate rollouts for every
positive/negative perturbation. Here each response and verifier reward is
sampled once. For a search basis `V=[v_1,...,v_q]`, ordinary teacher-forced
inference estimates directional token scores

\[
z_{igtj}=\frac{\log\pi_{a+\mu v_j}(y_{igt}\mid s_{igt})-
\log\pi_{a-\mu v_j}(y_{igt}\mid s_{igt})}{2\mu}.
\]

Those scores provide projected GRPO derivatives and a small token-local Fisher
matrix. FO-NPG solves the damped projected natural-gradient system and accepts
an update only when the fixed-rollout surrogate and empirical KL satisfy the
locked line-search rules. Directional probes repeat neither generation nor
reward evaluation.

vLLM sampler probabilities and Hugging Face policy probabilities are kept
separate. The PPO denominator is the frozen Hugging Face old policy, while the
detached `pi_old_HF / q_vLLM` term only corrects inference-engine mismatch.
This mirrors the pinned TRL contract and prevents backend numerical differences
from masquerading as a policy update.

## Current evidence

Two earlier experiment families are retained under `artifacts/` for
reproducibility, but are **not headline efficacy results**:

- `pilot_residual_core_v3/` contains long H100 runs over a custom 256-scalar
  residual-core policy. Those runs established vLLM/FlashAttention throughput,
  rollout provenance, zero backward calls, and memory/timing instrumentation,
  but they are not normal LoRA GRPO and used an incorrect old-policy denominator.
- `reference_trl_lora_overfit25/` is a genuine upstream-TRL all-layer LoRA
  engineering run. It verifies the 1,089,536-parameter layout and that the
  optimizer moves it, but it did not improve its small validation set and did
  not use the final matched rollout/evaluation protocol.

No final BP-versus-FO efficacy claim is made until the corrected matched sweep
passes its gates and the locked final partition is evaluated once.

## H100 setup

The project uses `uv`. FlashAttention is installed from the requested
checksum-pinned prebuilt wheel and is never compiled from source.

```bash
git clone https://github.com/anpaure/rl-no-backward.git
cd rl-no-backward
./scripts/install_standard_grpo_h100.sh
```

The installer pins Torch 2.11/cu130, Transformers 5.15, vLLM 0.22, TRL 1.10,
PEFT 0.20, and FlashAttention 2.8.3. Run local tests with:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

Print or run the matched learning gate:

```bash
PYTHONPATH=src .venv-standard-grpo/bin/python \
  -m rl_no_backward.matched_lora_runner \
  --config configs/gsm8k_matched_lora_overfit_gate.yaml \
  --print-plan

PYTHONPATH=src .venv-standard-grpo/bin/python \
  -m rl_no_backward.matched_lora_runner \
  --config configs/gsm8k_matched_lora_overfit_gate.yaml \
  --output /path/outside/the/checkout/matched-lora-gate
```

Each method/seed runs in a fresh Python process. Raw JSONL, W&B offline runs,
selected LoRA checkpoints, sampled completions, immutable data/schedule/state
receipts, and validation results are written below the output directory.

## Plots and W&B

Append-only JSONL is the plotting source of truth. Matplotlib writes PNG and
PDF learning curves, selected-checkpoint performance, compute/memory tradeoffs,
and optimizer diagnostics:

```bash
PYTHONPATH=src .venv-standard-grpo/bin/python \
  -m rl_no_backward.plotting /path/to/results/raw artifacts/figures
```

W&B defaults to offline mode on the shared H100. After authenticating to the
intended entity, runs can be uploaded without rerunning training:

```bash
wandb sync /path/to/results/wandb/offline-run-*
```

The final measured tables and interpretation will replace the clearly marked
pending sections in [`REPORT.md`](REPORT.md) after the matched multi-seed run.

## Scope

This is a small-model feasibility study. A no-backward method may reduce
activation memory and fit inference-only infrastructure, but with `q`
two-sided directions it generally performs many more forward evaluations than
reverse-mode GRPO. The experiment therefore reports reward, exact accuracy,
environment samples, scored tokens, synchronized wall time (including LoRA
sync), peak memory, KL, update acceptance, and estimator fidelity rather than
assuming forward-only is faster.
