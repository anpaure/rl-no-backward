# Matched LoRA full-data gate (100 steps, seed 0)

This artifact is the completed one-seed, 100-step full-data gate for the
corrected comparison. It is evidence that both optimizers learn under the
matched protocol; it is not the final multi-seed result.

## Contract

- Source commit: `69f1fcd21e8d94427dd769c7c03791021602627a`
- Model: pinned `Qwen/Qwen2.5-1.5B-Instruct`
- Task: full-difficulty GSM8K, exact numeric-match reward
- Policy: identical PEFT LoRA on `q_proj` and `v_proj` in all 28 blocks,
  rank 8 / alpha 16, exactly 1,089,536 FP32 policy parameters
- Budget: 100 updates, 8 prompts x 8 completions = 6,400 environment
  responses per trained method
- Backend: vLLM 0.22, BF16, FlashAttention-2, CUDA graphs
- Evaluation: the committed 256-example development partition; the locked
  679-example final partition was not accessed

## Result

| Method | Selected step | Selected dev exact | Gain vs base | Final-step dev | Training phase | Peak allocated | Peak reserved | Backward calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | 65.625% | - | 65.625% | - | - | - | 0 |
| BP-GRPO | 60 | 70.703% | +5.078 pp | 69.141% | 619.88 s | 29.55 GiB | 35.83 GiB | 800 |
| FO-NPG | 20 | 69.141% | +3.516 pp | 68.750% | 1,947.72 s | 25.14 GiB | 31.11 GiB | 0 |

FO-NPG finished 1.5625 percentage points below BP-GRPO, used 3.142x the
training-phase time, and saved 4.405 GiB of peak allocated memory. It accepted
93 of 100 trust-region proposals. Both methods consumed exactly 6,400
environment responses.

The forward-only direction is a predeclared finite-scale BF16 approximation,
not an exact infinitesimal gradient: the non-updating real-model oracle passed
at cosine 0.912 and relative L2 error 0.413 for `q=8`, `mu=10`.

## Contents

- `raw/`: append-only per-step JSONL, including rollout and policy digests
- `checkpoints/`: selected LoRA-only checkpoints
- `selection/`: selected-step and state-digest receipts
- `learning_gate/`: structural and observed-learning receipts
- `diagnostics/`: TRL differential and projected-gradient oracle evidence
- `schedules/`: immutable prompt and rollout schedules
- `samples/`: development predictions for the selected policies
- `wandb/`: three clean offline W&B runs
- `artifact_validation.json`: `complete`, 3/3 runs, no warnings or errors
- `figures/training_reward_and_dev_accuracy.png`: raw and 10-step-smoothed
  rollout reward plus development/selection trajectories
- `figures/grpo_surrogate_objective_change.png`: comparable zero-centered
  post-minus-pre GRPO surrogate change (not an absolute loss)
- `figures/optimizer_diagnostics.png`: gradient/score norm, policy KL, step
  norm, and update acceptance diagnostics
- `figures/compute_memory_tradeoffs.png`: measured time/memory comparison
- `figures/paired_method_comparisons.png`: paired FO-minus-BP deltas

Training phase is exactly policy synchronization + rollout/HF-old-policy
scoring + optimizer time. It excludes development evaluation. Peak GPU memory
is the combined in-process Hugging Face + vLLM CUDA allocator scope.
