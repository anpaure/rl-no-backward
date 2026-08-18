# Standard LoRA GRPO vs. forward-only RL on GSM8K

## Status

The corrected matched experiment is in its predeclared integration and
learning-gate phase. Final results are intentionally pending.

The headline comparison is now ordinary GRPO versus strict forward-only NPG
over the **same** standard PEFT LoRA policy: Qwen2.5-1.5B-Instruct, `q_proj` and
`v_proj` in all 28 blocks, rank 8, alpha 16, exactly 1,089,536 policy
parameters. Both methods use the same initialization artifact, prompt and
rollout seeds, exact GSM8K reward, vLLM/FlashAttention-2 generation backend,
Hugging Face old-policy scores, response budget, and development examples.

The frozen-batch GRPO objective is differentially checked against upstream TRL
1.10. The no-backward optimizer estimates the same objective's projected
derivatives by symmetric inference-only rescoring and uses a token-local
Fisher/KL trust region. A real-model finite-difference-versus-backprop
diagnostic must pass before training results are accepted.

## Pilot evidence retained, not promoted

The repository preserves two useful but non-headline artifact sets:

| Artifact | What it establishes | Why it is not the answer |
|---|---|---|
| `pilot_residual_core_v3` | H100/vLLM/FA2 performance, provenance, zero FO backward calls, long-run telemetry | Custom 256-scalar residual adapter and an incorrect PPO old-policy denominator |
| `reference_trl_lora_overfit25` | Genuine all-layer standard LoRA and upstream TRL optimizer movement | Small engineering run, unmatched evaluation/rollout protocol, no validation improvement |

The completed residual-core runs all selected the initialization checkpoint;
they are not evidence that normal GRPO fails on GSM8K. They are explicitly
quarantined from the corrected efficacy comparison.

## Locked evaluation protocol

Prior experiments exposed 384 distinct official-test IDs. They are excluded.
The remaining 935 IDs were deterministically partitioned before the corrected
run into 256 development IDs and 679 locked final IDs. Training processes load
only the committed development source indices. The locked partition will be
evaluated once after methods, hyperparameters, steps, and seeds are frozen.

## Pending result table

The final report will include, for the base model, BP-GRPO, and FO-NPG:

- selected-checkpoint development and locked-test exact accuracy;
- paired multi-seed uncertainty;
- rollout reward and zero-advantage trajectories;
- environment samples and generated/scored tokens;
- policy-sync, rollout/rescore, optimizer, evaluation, and total wall time;
- peak allocated/reserved GPU memory;
- empirical KL, accepted-step rate, and estimator fidelity;
- explicit BP backward-call and FO zero-backward-call receipts.

Matplotlib PNG/PDF figures and `summary.csv` will be generated from the raw
append-only JSONL after artifact validation passes. No placeholder number in
this report should be interpreted as a completed result.
