# Matched optimizer power traces

This artifact contains one matched BP-GRPO optimizer update and one strictly
backward-free FO-NPG optimizer update on Qwen2.5-1.5B-Instruct. Both methods
start from the same 1,089,536-parameter LoRA state and consume the same 32
responses, rewards, behavior log-probabilities, and HF old-policy scores.

## Result

| Metric | BP-GRPO | FO-NPG | FO / BP |
|---|---:|---:|---:|
| Backward calls | 4 | 0 | — |
| Physical teacher-forced forward calls | 8 | 38 | 4.75x |
| Logical policy evaluations | 2 | 19 | 9.50x |
| Optimizer duration | 1.033 s | 5.226 s | 5.06x |
| NVML mean power | 187.6 W | 317.2 W | 1.69x |
| NVML peak power | 305.4 W | 352.1 W | 1.15x |
| NVML raw energy over optimizer interval | 193.7 J | 1,657.6 J | 8.56x |
| Baseline-adjusted NVML dynamic energy | 86.8 J | 1,095.5 J | 12.62x |
| ChipWhisperer optimizer AC RMS | 0.01549 | 0.01067 | 0.69x |

The backward-free update is not faster in this configuration. Its q=8
two-sided estimator and line search require many full-model forward rescoring
passes, so it remains active about five times longer and uses about 8.6 times
the raw measured GPU energy despite executing no reverse-mode pass.

## Capture protocol

- ChipWhisperer Husky Plus, true streaming mode, 20 s at 100 kHz
  (2,000,000 samples per method), 12 bit, 10 dB gain.
- Normal/slow FIFO is enforced after arm; the corrupt fast-FIFO path is
  disabled and read back before a force trigger, with a 1 ms settle.
- A 1 s pre-optimizer baseline precedes each update.
- The captured region is only the optimizer update; rollout generation,
  model loading, compilation, and evaluation are excluded.
- Both SideCapture records passed finite-value, expected-length, variance,
  flatline, ADC clipping, and annotation-bounds checks with zero ADC errors.
- Source commit: `a76a1ee03db10d52a53470b01eb1861ccbfb9ddb`.

The primary ChipWhisperer channel is an external AC-coupled, uncalibrated
`normalized_adc` power proxy. Its amplitude is not watts and must not be
integrated as joules. Timestamped NVML is included as an auxiliary channel for
the watt and joule summaries above.

This is a descriptive n=1 comparison, not a repeat-based estimate: there is no
confidence interval, and transient clock/power state can affect the numerical
ratios. The pair receipt nevertheless cryptographically verifies identical
inputs and initialization, one healthy capture per method, BP backward calls
greater than zero, and FO backward calls equal to zero.

## Files

- `power_pair.json`: pair-level validity and matched digests.
- `bp_grpo/` and `fo_npg/`: immutable result receipts and raw SideCapture
  stores, including the full ChipWhisperer and NVML arrays.
- `figures/power_trace_comparison.png` and `.pdf`: aligned comparison plot.
- `figures/power_trace_summary.csv`: plotted numerical summaries.
