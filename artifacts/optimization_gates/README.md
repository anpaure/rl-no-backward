# H100 optimization gates

These artifacts are short correctness and performance gates for the accelerated
trainer. They are not included in the final accuracy comparison.

Both successful gates use Qwen2.5-1.5B-Instruct, HF FlashAttention-2,
vLLM 0.22 forced to its FA2 kernel, the rank-8 adapters in the final four
blocks, frozen-prefix replay, selective response logits, and the same grouped
GSM8K rollout construction as training. W&B was disabled for these disposable
gates; append-only JSONL is retained.

## Results

- `vllm_fa2_eager_gate`: vLLM eager mode, two BP-GRPO and two FO-NPG
  updates. BP optimizer time was 0.16--0.27 seconds per step; q=8 FO-NPG was
  0.57--0.79 seconds and executed no backward pass. Rollout plus old-policy
  scoring took 3.2--4.8 seconds.
- `vllm_fa2_graph_nonzero_gate`: compiled vLLM with CUDA graphs, three updates
  per method. Rollout plus old-policy scoring fell to 1.08--1.83 seconds; BP
  optimizer time was 0.16--0.28 seconds and FO-NPG was 0.53--0.70 seconds.
  Step 2 changed both policies, and step 3 generated and passed the HF/vLLM
  log-probability gate from those nonzero policy digests. This proves the
  captured graph observes synchronized adapter-core updates rather than stale
  constants.
- `vllm_fa2_graph_inproc_final_gate`: the selected production shape: fast FA2
  and CUDA graphs, deterministic in-process scheduling, BP microbatch 4, and
  fused FO probes at two directions by four examples. BP rollout+optimizer
  phases were 1.61--2.05 seconds; FO-NPG phases were 1.75--2.23 seconds. The
  two methods' first-rollout token, mask, behavior-log-probability, and combined
  SHA-256 identities matched exactly. The production metadata correctly records
  that memory is the combined HF+vLLM single-process allocator and that no
  insecure callable serialization is enabled.
- `vllm_fa2_graph_batch_invariant_gate`: vLLM's deterministic-kernel mode also
  produced exact matched first rollouts, but rollout phases rose to 2.36--4.03
  seconds. It is retained as negative performance evidence and is not enabled
  in the long run.
- `vllm_fa2_graph_inproc_bp16_gate`: increasing the BP scoring microbatch from
  4 to 16 reduced the optimizer by only about 0.02--0.04 seconds while raising
  the combined allocator peak from roughly 32 GiB reserved to 54 GiB. The
  final config therefore keeps microbatch 4.

Across the graph-mode training records, HF/vLLM absolute token-log-probability
differences had means `0.0064`--`0.0120`, p99 values `0.127`--`0.144`, and
maxima `0.289`--`0.308`. The actual vLLM behavior log probabilities remain the
GRPO denominator; the HF score is logged as a cross-engine numerical audit.

## Attention benchmark

The JSON files under `benchmarks/` deliberately preserve a failed strict
equivalence gate. Identical base-weight hashes showed that Transformers'
manual BF16 eager attention diverged from the first layer, while SDPA and
FlashAttention were much closer. Sampled completion timings were also not
compute-equivalent once outputs diverged. Consequently, the optimized sweep
uses FlashAttention consistently and does not use eager as its numerical
oracle or quote those sampled timings as a speedup.

The gate metadata records a dirty source tree based on commit `fb3473d` because
these runs were executed while the acceleration patch was being developed.
The locked long sweep runs from a clean published commit in a separate artifact
directory; none of these short gates enter its accuracy aggregation.
