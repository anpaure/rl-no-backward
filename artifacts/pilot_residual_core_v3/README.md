# Residual-core H100 pilot (not a headline efficacy result)

This directory preserves the long-running H100 pilot requested during
development. It contains append-only JSONL, selected checkpoints, samples, and
W&B offline runs for a frozen Qwen2.5-1.5B base with four custom rank-8
residual cores (256 optimized scalars).

Use this evidence only for implementation and systems questions: vLLM/FA2
throughput, policy-state and rollout provenance, forward/backward call
accounting, wall time, memory, and proof that the forward-only path recorded
zero backward calls. The sweep was interrupted during FO-NPG seed 1 and is
therefore incomplete.

Do **not** use these files as the normal-GRPO-versus-gradient-free efficacy
comparison. The policy was not standard LoRA, and the pilot incorrectly put
vLLM sampler probabilities directly in the PPO denominator. Every completed
trained run selected checkpoint 0. The corrected matched standard-LoRA
experiment lives in the repository's `matched_lora_*` modules and configs.
