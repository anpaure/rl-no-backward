# RL Without Backward Passes

Can a small language model learn from reinforcement learning without executing
a backward pass? This repository compares ordinary reverse-mode GRPO with
strict inference-only parameter updates on a small, reproducible GSM8K RLVR
benchmark.

The primary backward-free method samples each rollout group once, symmetrically
perturbs a tiny adapter, rescoring the *same stored completions* under each
perturbed policy, reconstructs a projected policy gradient and token-level
empirical Fisher matrix, then takes a KL-constrained natural-gradient step.
Rewards and environment interaction are never repeated for the directional
probes.

## Experiment

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, frozen in BF16.
- **Adapter:** four activation-PCA residual cores in the last four blocks,
  rank 8; only 256 scalars are trainable.
- **Task:** full-difficulty GSM8K with deterministic final-number checking.
  The training reward is exact match plus at most 0.1 bounded numeric-proximity
  shaping; exact match remains the reported metric.
- **Data/budget:** 512 official-train examples, 96 disjoint validation examples,
  and 300 optimizer steps (4,800 sampled responses per trained run). A fresh
  256-example official-test subset is evaluated only after a checkpoint is
  selected by validation accuracy; all pilot-test IDs are excluded.
- **Methods:** frozen base, BP-GRPO on the same adapter, forward-only projected
  PG, forward-only natural PG, and a history-subspace NPG heuristic.
- **Fairness:** identical adapter initialization, examples, prompt schedule,
  rollout seeds, group size, verifier, and validation selection rule.

The history-subspace method is deliberately labeled as a heuristic. It is not
the document's full independent cross-sketch covariance estimator. FO-NPG is
the primary backward-free comparison.

## Why not difference rewards directly?

Return-difference evolution strategies must regenerate rollouts for every
positive/negative perturbation, and their noise scales poorly as the probe
radius shrinks. Here, each response and reward is sampled once. Ordinary
teacher-forced inference provides the directional action-log-probability score
needed by the policy-gradient identity.

For a search basis \(V=[v_1,\ldots,v_q]\), the implementation estimates

\[
z_{igtj}=\frac{\log\pi_{a+\mu v_j}(y_{igt}\mid s_{igt})-
\log\pi_{a-\mu v_j}(y_{igt}\mid s_{igt})}{2\mu},
\]

then forms a length-normalized projected gradient and the token-level Fisher
that matches the constrained KL geometry:

\[
g_V=\frac1{BG}\sum_{ig}A_{ig}\frac1{T_{ig}}\sum_t z_{igt},\qquad
F_V=\frac1{BG}\sum_{ig}\frac1{T_{ig}}\sum_t z_{igt}z_{igt}^{\top}.
\]

FO-NPG solves \((F_V+\lambda I)u=g_V\), proposes
\(\Delta a=\alpha Vu\), and uses inference-only line search to enforce the KL
budget and improve the stored-rollout clipped surrogate.

## Reproduce

The project is locked with `uv.lock` and all published training was executed on
one NVIDIA H100 PCIe.

```bash
git clone https://github.com/anpaure/rl-no-backward.git
cd rl-no-backward
uv sync --extra dev
uv run pytest -q
uv run rl-no-backward gsm8k \
  --config configs/gsm8k_final.yaml \
  --output artifacts/final_gsm8k
uv run python -m rl_no_backward.plotting \
  artifacts/final_gsm8k artifacts/figures
```

For the optimized H100 path, the setup script creates a separate vLLM 0.22
runtime and installs the matching, checksum-verified FlashAttention-2 wheel
from the requested prebuilt-wheel repository; it does not compile
FlashAttention from source:

```bash
./scripts/install_h100_fastpath.sh
PYTHONPATH=src .venv-vllm/bin/python -m rl_no_backward.cli gsm8k \
  --config configs/gsm8k_optimized_final.yaml \
  --output artifacts/final_gsm8k_optimized
```

The optimized trainer can cache the frozen first 24 Qwen blocks once per
stored rollout and replay only the adapted final four blocks for old-policy
scoring, BP updates, forward-only probes, and line search. It also asks Qwen's
LM head to materialize logits only at response-prediction positions. Both
fast paths are opt-in and fall back with explicit telemetry when the model
structure is unsupported.

The headline config uses vLLM's deterministic in-process scheduler, preserving
the fast FlashAttention-2/CUDA-graph kernels while removing multiprocessing
request-order drift. Every training rollout also carries SHA-256 identities for
its tokens, masks, behavior log probabilities, and sampling seed. HF/vLLM
behavior-policy agreement is tolerance-gated at the mean, p99, and maximum
token-log-probability levels rather than requiring bitwise equality across the
two different inference implementations. vLLM's optional batch-invariant
kernels remain supported, but are disabled in the headline config because the
H100 gate measured a substantial throughput penalty. See the official
[vLLM reproducibility](https://docs.vllm.ai/en/v0.22.0/usage/reproducibility/)
and [batch-invariance](https://docs.vllm.ai/en/v0.22.0/features/batch_invariance/)
notes for that tradeoff.

On an H100 host that already provides a compatible CUDA PyTorch build, a
system-site-packages environment avoids downloading another multi-gigabyte
wheel:

```bash
uv venv --system-site-packages --python /usr/bin/python3
uv pip install 'transformers>=4.48,<6' 'datasets>=3,<5' \
  'matplotlib>=3.9' 'wandb>=0.19,<1' 'pytest>=8' 'ruff>=0.9'
uv pip install --no-deps -e .
```

Run the real-model numerical audit separately:

```bash
uv run python -m rl_no_backward.gsm8k_diagnostics \
  --output artifacts/diagnostics/optimized_hf_exact/gsm8k_fd.json \
  --max-tokens 512 --directions 8 --frozen-prefix-scoring
```

## W&B and raw evidence

Every method/seed logs metrics and a run-data artifact through W&B. Configs use
offline mode by default so credentials are never assumed on a shared machine.
After authenticating to the intended entity, upload them with:

```bash
wandb sync artifacts/final_gsm8k_optimized/wandb/offline-run-*
```

Append-only JSONL is the plotting source of truth, so all figures can be
regenerated without W&B. The final report and measured results are in
[`REPORT.md`](REPORT.md) and `artifacts/figures/` once the locked sweep is
complete.

The completed unoptimized base and BP-GRPO seed-0 run is preserved separately
under [`artifacts/reference_eager_gsm8k`](artifacts/reference_eager_gsm8k).
Its README labels it as a regression/performance reference and records the
interrupted seed that was deliberately excluded.

## Scope

This is a small-model feasibility study, not evidence that forward-only RL is
competitive for frontier-scale reasoning post-training. It isolates optimizer
behavior under a severe 256-parameter adapter bottleneck and reports sample,
compute, time, memory, KL, estimator, and reward diagnostics explicitly.
