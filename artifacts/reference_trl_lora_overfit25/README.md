# Upstream TRL standard-LoRA engineering pilot

This directory preserves a 25-step upstream TRL 1.10 run on
Qwen2.5-1.5B-Instruct with conventional PEFT LoRA (`q_proj` and `v_proj`, rank
8, alpha 16, all 28 blocks). Its 112 LoRA tensors contain exactly 1,089,536
FP32 parameters, and the optimizer changed the adapter state.

This is a useful reference implementation artifact, not the final baseline.
It used a small engineering split and stock TRL synchronization/evaluation,
did not certify the requested vLLM FlashAttention-2 path, and validation exact
accuracy did not improve between step 0 and step 25. The corrected comparison
uses a shared matched runner so BP-GRPO and FO-NPG receive the same policy
initialization, prompt schedule, rollouts, old-policy scores, reward, budget,
and evaluation backend.
