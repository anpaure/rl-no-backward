# Eager GSM8K reference run

This directory preserves the completed portion of the original, unoptimized
H100 run from source commit `fb3473d9b88434b591b18bff88ccad42285abdc1`.
It is a performance and regression reference, not the final multi-method
scientific sweep.

## Complete artifacts included

- Frozen-base evaluation (`base`, seed 0).
- BP-GRPO training (`bp_grpo`, seed 0): 300 optimizer steps and 4,800 sampled
  responses.
- Validation-selected checkpoints, raw JSONL metrics, generated samples, and
  the two matching W&B offline bundles.
- The exact run configuration and provenance metadata.

The BP-GRPO run selected step 250 at validation exact accuracy `0.5208333`,
up from the frozen model's `0.4791667`. Its locked official-test exact
accuracy was `0.3359375`, tied with the frozen model. The full BP trial took
`3348.25` seconds (55.80 minutes), including final evaluation, and reported a
peak PyTorch allocation of `16,343,133,696` bytes.

## Deliberately excluded

The original configured sweep continued into BP-GRPO seed 1, but it was
stopped after step 4 so optimization work could proceed. That partial JSONL
and its unfinished W&B run are intentionally not included here. No FO method
completed in this reference run.

The retained W&B runs are:

- `offline-run-20260818_141259-t99vfbm9` (base)
- `offline-run-20260818_141655-8xai4rfw` (BP-GRPO seed 0)

Both retained W&B debug logs record a clean finish. The final optimized sweep
uses a separate artifact directory and must pass the repository's complete
artifact validator before results are reported.
