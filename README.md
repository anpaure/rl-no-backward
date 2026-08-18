# RL Without Backward Passes

This repository compares a conventional backpropagation GRPO baseline with
strictly forward-only reinforcement-learning optimizers for a frozen small
language model.  The forward-only methods reuse one rollout group, estimate
directional action-log-probability derivatives with symmetric inference passes,
and update a tiny activation-calibrated adapter in a random or learned subspace.

The initial benchmark is intentionally small: a deterministic one-token
checksum task with exact rewards and a constrained ten-digit action space.  It
is a proof-of-concept for optimizer behavior, not evidence about open-ended
reasoning RL.

Full reproduction commands, results, figures, and conclusions will be added
after the locked multi-seed run.

