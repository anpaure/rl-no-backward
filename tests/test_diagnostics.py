from __future__ import annotations

import json
from pathlib import Path

from rl_no_backward.diagnostics import (
    DiagnosticReport,
    audit_forward_only_source,
    finite_difference_vs_autograd,
    fisher_kl_sanity,
    run_diagnostics,
    write_diagnostic_results,
)


def test_central_finite_difference_matches_exact_autograd() -> None:
    result = finite_difference_vs_autograd()

    assert result.passed
    assert result.forward_calls == 2 * result.directions
    assert result.max_absolute_error < result.absolute_tolerance
    assert result.mean_absolute_error < result.absolute_tolerance / 5


def test_forward_only_source_has_no_reverse_mode_calls() -> None:
    result = audit_forward_only_source()

    assert result.passed, result.violations
    assert result.violations == ()
    assert Path(result.source_path).name == "forward_only.py"


def test_source_audit_finds_backward_and_aliased_autograd(tmp_path: Path) -> None:
    source = tmp_path / "bad_forward_only.py"
    source.write_text(
        """from torch.autograd import grad as exact_grad
import torch

x.backward()
exact_grad(x, y)
torch.autograd.grad(x, y)
""",
        encoding="utf-8",
    )

    result = audit_forward_only_source(source)

    assert not result.passed
    assert [violation.call for violation in result.violations] == [
        "x.backward",
        "torch.autograd.grad",
        "torch.autograd.grad",
    ]


def test_fisher_solve_and_kl_helpers_are_numerically_sane() -> None:
    result = fisher_kl_sanity()

    assert result.passed
    assert result.fisher_minimum_eigenvalue >= 0.0
    assert result.solve_residual_norm <= result.tolerance
    assert result.zero_kl_absolute_error <= result.tolerance
    assert result.perturbed_kl > 0.0
    assert result.manual_kl_absolute_error <= result.tolerance


def test_diagnostic_writer_emits_complete_json(tmp_path: Path) -> None:
    report = run_diagnostics()
    output = write_diagnostic_results(tmp_path / "nested" / "diagnostics.json", report)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert isinstance(report, DiagnosticReport)
    assert output.exists()
    assert payload["passed"] is True
    assert payload["directional_derivative"]["forward_calls"] == 8
    assert payload["source_audit"]["violations"] == []
    assert payload["fisher_kl"]["perturbed_kl"] > 0.0
