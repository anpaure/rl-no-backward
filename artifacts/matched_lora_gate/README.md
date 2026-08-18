# Matched standard-LoRA learning gate

This is the first successful end-to-end gate for the corrected comparison. It
is a one-seed, 25-update engineering/learning gate, not the final multi-seed
headline result.

- Model: pinned `Qwen/Qwen2.5-1.5B-Instruct`
- Policy: identical PEFT LoRA on `q_proj` and `v_proj` in all 28 blocks,
  rank 8 / alpha 16, exactly 1,089,536 trainable scalars
- Task: GSM8K exact-match RLVR
- Budget: 8 prompts x 8 completions x 25 updates = 1,600 responses per
  trained method
- Backends: Hugging Face BF16 FlashAttention-2 scoring and vLLM 0.22
  FlashAttention-2 rollout/evaluation
- Source: commit `cc02cd0b96f8c1f330b321877cb5fe04dc0527a4`

The common development accuracy was 65.625% at initialization. Backprop GRPO
selected step 20 at 70.3125%; strict forward-only NPG selected step 25 at
68.3594%. The forward-only process recorded zero backward calls and accepted
13 of 25 proposed updates.

Synchronized training phases totaled 134.7 seconds for backprop GRPO and
484.4 seconds for forward-only NPG. Peak process CUDA allocation was 28.29 GiB
and 24.85 GiB, respectively. Thus this implementation trades lower activation
memory for roughly 3.6x more training-phase wall time; repeated forward probes
dominate the no-backward method.

`artifact_validation.json` reports a complete 3/3 bundle. The directory keeps
raw JSONL, selected checkpoints, samples, per-trial metadata, immutable rollout
receipts, offline W&B runs, and Matplotlib PNG/PDF figures.
