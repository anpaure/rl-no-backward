"""Numerical and source-level diagnostics for the forward-only optimizer.

These checks are intentionally small enough to run on CPU.  They provide
evidence for three claims that matter to the experiment:

* central finite differences recover true directional score derivatives;
* the forward-only implementation does not call reverse-mode autograd; and
* the damped Fisher solve and categorical-KL helper behave as expected.

``run_diagnostics`` returns structured dataclasses, while
``write_diagnostic_results`` persists the same information as JSON for an
experiment artifact or CI log.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from .common import categorical_kl
from .forward_only import (
    ForwardConfig,
    _solve_coordinates,
    directional_log_probability_scores,
)
from .model import ModelBundle, candidate_log_probs, parameter_vector


@dataclass(frozen=True)
class DirectionalDerivativeResult:
    """Agreement statistics for finite-difference and autograd derivatives."""

    passed: bool
    finite_difference_mu: float
    directions: int
    forward_calls: int
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True)
class SourceViolation:
    """One forbidden reverse-mode call found by the source audit."""

    line: int
    column: int
    call: str
    reason: str


@dataclass(frozen=True)
class SourceAuditResult:
    """Result of statically auditing a Python module for autograd calls."""

    passed: bool
    source_path: str
    violations: tuple[SourceViolation, ...]


@dataclass(frozen=True)
class FisherKLSanityResult:
    """Numerical checks for the damped Fisher solve and categorical KL."""

    passed: bool
    fisher_minimum_eigenvalue: float
    solve_residual_norm: float
    zero_kl_absolute_error: float
    perturbed_kl: float
    manual_kl_absolute_error: float
    tolerance: float


@dataclass(frozen=True)
class DiagnosticReport:
    """Complete lightweight diagnostic report."""

    directional_derivative: DirectionalDerivativeResult
    source_audit: SourceAuditResult
    fisher_kl: FisherKLSanityResult

    @property
    def passed(self) -> bool:
        return (
            self.directional_derivative.passed
            and self.source_audit.passed
            and self.fisher_kl.passed
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


class _ToyPolicy(nn.Module):
    """Tiny differentiable categorical policy with one adapter-like parameter."""

    def __init__(self, feature_count: int, action_count: int) -> None:
        super().__init__()
        self.adapter = nn.Module()
        self.adapter.register_parameter(
            "core", nn.Parameter(torch.empty(feature_count, action_count))
        )

    def forward(self, features: Tensor, use_cache: bool = False) -> SimpleNamespace:
        del use_cache
        logits = features @ self.adapter.core
        return SimpleNamespace(logits=logits.unsqueeze(1))


def _toy_bundle_and_inputs(
    *, seed: int, device: torch.device
) -> tuple[ModelBundle, dict[str, Tensor]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    policy = _ToyPolicy(feature_count=3, action_count=4).to(device)
    with torch.no_grad():
        policy.adapter.core.copy_(
            0.35
            * torch.randn(
                policy.adapter.core.shape,
                generator=generator,
                device=device,
                dtype=policy.adapter.core.dtype,
            )
        )
    features = torch.randn(5, 3, generator=generator, device=device)
    bundle = ModelBundle(
        model=policy,
        tokenizer=None,
        candidate_token_ids=torch.arange(4, device=device),
        adapter_names=["adapter.core"],
        device=device,
        model_name="diagnostic-toy-policy",
    )
    return bundle, {"features": features}


def _exact_directional_log_probability_scores(
    bundle: ModelBundle,
    encoded: dict[str, Tensor],
    basis: Tensor,
) -> Tensor:
    """Form the exact log-probability Jacobian-vector products with autograd."""

    parameters = bundle.trainable_parameters
    log_probs = candidate_log_probs(bundle, encoded)
    rows: list[Tensor] = []
    flattened = log_probs.reshape(-1)
    for index, value in enumerate(flattened):
        gradients = torch.autograd.grad(
            value,
            parameters,
            retain_graph=index + 1 < flattened.numel(),
        )
        gradient_vector = torch.cat([gradient.reshape(-1) for gradient in gradients])
        rows.append(gradient_vector @ basis)
    return torch.stack(rows).reshape(*log_probs.shape, basis.shape[1]).detach()


def finite_difference_vs_autograd(
    *,
    mu: float = 1e-3,
    directions: int = 4,
    seed: int = 2026,
    device: str | torch.device = "cpu",
    absolute_tolerance: float = 3e-4,
    relative_tolerance: float = 3e-3,
) -> DirectionalDerivativeResult:
    """Compare the production central-difference scores with exact autograd.

    The test policy has only twelve parameters, so the full output Jacobian is
    cheap to form.  This diagnostic deliberately uses autograd only as an
    external oracle; the implementation under test remains inference-only.
    """

    if mu <= 0:
        raise ValueError("mu must be positive")
    if directions <= 0:
        raise ValueError("directions must be positive")

    target_device = torch.device(device)
    bundle, encoded = _toy_bundle_and_inputs(seed=seed, device=target_device)
    center = parameter_vector(bundle).float()
    if directions > center.numel():
        raise ValueError(
            f"directions ({directions}) cannot exceed toy parameter count ({center.numel()})"
        )
    generator = torch.Generator(device=target_device).manual_seed(seed + 1)
    raw_basis = torch.randn(
        center.numel(),
        directions,
        generator=generator,
        device=target_device,
        dtype=center.dtype,
    )
    basis = torch.linalg.qr(raw_basis, mode="reduced").Q

    exact = _exact_directional_log_probability_scores(bundle, encoded, basis)
    approximate, forward_calls = directional_log_probability_scores(
        bundle=bundle,
        encoded=encoded,
        center=center,
        basis=basis,
        mu=mu,
    )

    difference = (approximate - exact).abs()
    relative = difference / exact.abs().clamp_min(1e-7)
    passed = bool(
        torch.allclose(
            approximate,
            exact,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )
    return DirectionalDerivativeResult(
        passed=passed,
        finite_difference_mu=float(mu),
        directions=int(directions),
        forward_calls=int(forward_calls),
        max_absolute_error=float(difference.max().item()),
        mean_absolute_error=float(difference.mean().item()),
        max_relative_error=float(relative.max().item()),
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                bound_name = imported.asname or imported.name.split(".", maxsplit=1)[0]
                aliases[bound_name] = imported.name if imported.asname else bound_name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                if imported.name == "*":
                    continue
                bound_name = imported.asname or imported.name
                aliases[bound_name] = f"{node.module}.{imported.name}"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            name = _dotted_name(value) if value is not None else None
            targets: Sequence[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = (node.target,)
            if name is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = name
    return aliases


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    components = name.split(".")
    seen: set[str] = set()
    while components[0] in aliases and components[0] not in seen:
        root = components[0]
        seen.add(root)
        components = aliases[root].split(".") + components[1:]
    return ".".join(components)


def _forbidden_call_reason(name: str) -> str | None:
    components = name.split(".")
    callable_name = components[-1]
    if callable_name == "backward":
        return "reverse-mode backward call"
    if callable_name == "grad" and (
        "autograd" in components or components[:-1] in (["torch", "func"], ["func"])
    ):
        return "reverse-mode gradient call"
    return None


def audit_forward_only_source(source_path: str | Path | None = None) -> SourceAuditResult:
    """Parse ``forward_only.py`` and report forbidden reverse-mode calls.

    The audit understands ordinary import aliases (including
    ``from torch.autograd import grad as ...``), ignores comments/docstrings,
    and reports exact source locations for CI failures.
    """

    path = (
        Path(source_path)
        if source_path is not None
        else Path(__file__).with_name("forward_only.py")
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    aliases = _import_aliases(tree)
    violations: list[SourceViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        raw_name = _dotted_name(node.func)
        if raw_name is None:
            continue
        resolved_name = _resolve_alias(raw_name, aliases)
        reason = _forbidden_call_reason(resolved_name)
        if reason is not None:
            violations.append(
                SourceViolation(
                    line=node.lineno,
                    column=node.col_offset,
                    call=resolved_name,
                    reason=reason,
                )
            )
    violations.sort(key=lambda violation: (violation.line, violation.column))
    return SourceAuditResult(
        passed=not violations,
        source_path=str(path.resolve()),
        violations=tuple(violations),
    )


def fisher_kl_sanity(*, tolerance: float = 1e-10) -> FisherKLSanityResult:
    """Check a PSD empirical Fisher solve and two categorical KL identities."""

    score_rows = torch.tensor(
        [[1.0, -0.5], [0.25, 1.5], [-0.75, 0.5], [1.25, 0.75]],
        dtype=torch.float64,
    )
    fisher = score_rows.T @ score_rows / score_rows.shape[0]
    gradient = torch.tensor([0.4, -0.3], dtype=torch.float64)
    config = ForwardConfig(method="fo_npg", fisher_damping=0.2)
    coordinates = _solve_coordinates(gradient, fisher, config)
    scale = (torch.trace(fisher) / fisher.shape[0]).clamp_min(1e-8)
    regularized = fisher + config.fisher_damping * scale * torch.eye(
        fisher.shape[0], dtype=fisher.dtype
    )
    residual = regularized @ coordinates - gradient
    minimum_eigenvalue = torch.linalg.eigvalsh(fisher).min()

    old_logits = torch.tensor([[0.2, -0.3, 0.8], [1.0, 0.1, -0.4]], dtype=torch.float64)
    new_logits = old_logits + torch.tensor(
        [[0.1, -0.2, 0.05], [-0.15, 0.2, 0.1]], dtype=torch.float64
    )
    old_log_probs = old_logits.log_softmax(dim=-1)
    new_log_probs = new_logits.log_softmax(dim=-1)
    zero_kl = categorical_kl(old_log_probs, old_log_probs)
    perturbed_kl = categorical_kl(old_log_probs, new_log_probs)
    manual_kl = (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum(dim=-1).mean()

    residual_norm = float(residual.norm().item())
    zero_kl_error = abs(float(zero_kl.item()))
    perturbed_kl_value = float(perturbed_kl.item())
    manual_error = abs(float((perturbed_kl - manual_kl).item()))
    minimum_eigenvalue_value = float(minimum_eigenvalue.item())
    passed = (
        minimum_eigenvalue_value >= -tolerance
        and residual_norm <= tolerance
        and zero_kl_error <= tolerance
        and perturbed_kl_value > 0.0
        and manual_error <= tolerance
    )
    return FisherKLSanityResult(
        passed=passed,
        fisher_minimum_eigenvalue=minimum_eigenvalue_value,
        solve_residual_norm=residual_norm,
        zero_kl_absolute_error=zero_kl_error,
        perturbed_kl=perturbed_kl_value,
        manual_kl_absolute_error=manual_error,
        tolerance=float(tolerance),
    )


def run_diagnostics(
    *,
    forward_only_source: str | Path | None = None,
) -> DiagnosticReport:
    """Run all cheap diagnostics and return a structured report."""

    return DiagnosticReport(
        directional_derivative=finite_difference_vs_autograd(),
        source_audit=audit_forward_only_source(forward_only_source),
        fisher_kl=fisher_kl_sanity(),
    )


def write_diagnostic_results(
    destination: str | Path,
    report: DiagnosticReport | None = None,
    *,
    forward_only_source: str | Path | None = None,
) -> Path:
    """Write a diagnostic report as deterministic, human-readable JSON."""

    result = (
        report if report is not None else run_diagnostics(forward_only_source=forward_only_source)
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON artifact path")
    arguments = parser.parse_args(argv)
    report = run_diagnostics()
    if arguments.output is not None:
        write_diagnostic_results(arguments.output, report)
        print(arguments.output)
    else:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the callable API
    raise SystemExit(main())
