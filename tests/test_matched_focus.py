from __future__ import annotations

import ast
import inspect
from dataclasses import replace

import pytest
import torch

import rl_no_backward.matched_focus as focus_module
from rl_no_backward.matched_focus import (
    CanonicalLoRAPartition,
    CrossSketchCovarianceEMA,
    FocusCovarianceConfig,
    MatchedFocusState,
    build_focus_probe_plan,
    canonical_lora_partition,
    requires_b_only_bootstrap,
    symmetric_cross_sketch_action,
)
from rl_no_backward.standard_lora import LoRAParameterLayout


def _layout() -> LoRAParameterLayout:
    names = (
        "base.layers.0.q_proj.lora_A.default.weight",
        "base.layers.0.q_proj.lora_B.default.weight",
        "base.layers.0.v_proj.lora_A.default.weight",
        "base.layers.0.v_proj.lora_B.default.weight",
    )
    shapes = ((2, 3), (4, 2), (2, 3), (5, 2))
    return LoRAParameterLayout(
        names=names,
        shapes=shapes,
        numels=tuple(torch.Size(shape).numel() for shape in shapes),
        dtypes=("float32",) * len(names),
    )


def _partition() -> CanonicalLoRAPartition:
    return canonical_lora_partition(_layout())


def test_canonical_partition_pairs_and_round_trips_interleaved_layout() -> None:
    partition = _partition()
    assert partition.full_dimension == 30
    assert partition.a_dimension == 12
    assert partition.b_dimension == 18
    assert len(partition.pair_keys) == 2

    full = torch.arange(partition.full_dimension, dtype=torch.float32)
    a_values = partition.gather(full, "A")
    b_values = partition.gather(full, "B")
    torch.testing.assert_close(a_values, torch.cat([full[0:6], full[14:20]]))
    torch.testing.assert_close(b_values, torch.cat([full[6:14], full[20:30]]))
    torch.testing.assert_close(partition.merge(a_values, b_values), full)

    full_matrix = torch.stack([full, -full], dim=1)
    torch.testing.assert_close(
        partition.merge(
            partition.gather(full_matrix, "A"),
            partition.gather(full_matrix, "B"),
        ),
        full_matrix,
    )
    assert partition.layout_digest == _partition().layout_digest


def test_partition_rejects_noncanonical_or_unpaired_lora_layout() -> None:
    layout = _layout()
    with pytest.raises(ValueError, match="canonical lexicographic"):
        canonical_lora_partition(
            replace(
                layout,
                names=tuple(reversed(layout.names)),
                shapes=tuple(reversed(layout.shapes)),
                numels=tuple(reversed(layout.numels)),
                dtypes=tuple(reversed(layout.dtypes)),
            )
        )
    with pytest.raises(ValueError, match="paired A/B"):
        canonical_lora_partition(
            LoRAParameterLayout(
                names=(layout.names[0],),
                shapes=(layout.shapes[0],),
                numels=(layout.numels[0],),
                dtypes=("float32",),
            )
        )
    bad_shapes = list(layout.shapes)
    bad_shapes[1] = (4, 3)
    bad_numels = list(layout.numels)
    bad_numels[1] = 12
    with pytest.raises(ValueError, match="inconsistent ranks"):
        canonical_lora_partition(
            replace(layout, shapes=tuple(bad_shapes), numels=tuple(bad_numels))
        )


def test_b_only_bootstrap_detects_standard_lora_zero_b() -> None:
    partition = _partition()
    center = torch.randn(partition.full_dimension)
    center = partition.embed(partition.gather(center, "A"), "A")
    assert requires_b_only_bootstrap(center, partition)
    changed = center + partition.embed(torch.full((partition.b_dimension,), 1.0e-5), "B")
    assert not requires_b_only_bootstrap(changed, partition)
    assert requires_b_only_bootstrap(changed, partition, tolerance=1.0e-4)
    with pytest.raises(ValueError, match="non-negative"):
        requires_b_only_bootstrap(center, partition, tolerance=-1.0)


def test_raw_gaussian_cross_sketch_has_population_gradient_outer_product() -> None:
    # Two raw isotropic scouts and independent objective halves give
    # E[(d1 z1)(d2 z2)^T] = g g^T.  Exercise the matrix-free symmetric action
    # used by the implementation rather than constructing state covariance.
    generator = torch.Generator().manual_seed(20260819)
    gradient = torch.tensor([0.7, -0.4, 0.2])
    vectors = torch.eye(gradient.numel())
    accumulated = torch.zeros(gradient.numel(), gradient.numel())
    samples = 120_000
    batch_size = 2_000
    for _ in range(samples // batch_size):
        first_directions = torch.randn(batch_size, gradient.numel(), generator=generator)
        second_directions = torch.randn(batch_size, gradient.numel(), generator=generator)
        # Independent zero-mean half-batch noise cancels in the cross moment;
        # using one noisy half twice would instead learn its second moment.
        first_batch_gradients = gradient + 0.3 * torch.randn(
            batch_size,
            gradient.numel(),
            generator=generator,
        )
        second_batch_gradients = gradient + 0.3 * torch.randn(
            batch_size,
            gradient.numel(),
            generator=generator,
        )
        first_derivatives = (first_directions * first_batch_gradients).sum(dim=1)
        second_derivatives = (second_directions * second_batch_gradients).sum(dim=1)
        first_sketches = first_derivatives[:, None] * first_directions
        second_sketches = second_derivatives[:, None] * second_directions
        for first, second in zip(first_sketches, second_sketches, strict=True):
            accumulated += symmetric_cross_sketch_action(first, second, vectors)
    estimate = accumulated / samples
    torch.testing.assert_close(estimate, torch.outer(gradient, gradient), atol=1.5e-2, rtol=4.0e-2)


def test_cross_sketch_ema_matches_dense_positive_rank_two_update() -> None:
    config = FocusCovarianceConfig(ema_decay=0.25, relative_eigenvalue_floor=0.0)
    state = CrossSketchCovarianceEMA(2, config)
    first = torch.tensor([2.0, -1.0])
    second = torch.tensor([0.5, 3.0])
    state.update(first, second)

    dense = (
        (1.0 - config.ema_decay) * 0.5 * (torch.outer(first, second) + torch.outer(second, first))
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(dense)
    positive = eigenvalues.clamp_min(0)
    expected = (eigenvectors * positive[None, :]) @ eigenvectors.T
    torch.testing.assert_close(state.dense_for_testing(), expected, atol=2.0e-6, rtol=2.0e-6)
    assert state.rank <= 2
    assert state.persistent_numel <= 2 * 2 + 2
    with pytest.raises(ValueError, match="production-sized"):
        CrossSketchCovarianceEMA(65).dense_for_testing()


def test_q8_bootstrap_is_deterministic_b_only_and_reconstructs_raw_derivatives() -> None:
    partition = _partition()
    state = MatchedFocusState(partition)
    first = build_focus_probe_plan(
        partition,
        state,
        torch.Generator().manual_seed(17),
        bootstrap_b_only=True,
    )
    second = build_focus_probe_plan(
        partition,
        state,
        torch.Generator().manual_seed(17),
        bootstrap_b_only=True,
    )
    torch.testing.assert_close(first.basis, second.basis, rtol=0, atol=0)
    torch.testing.assert_close(first.reconstruction, second.reconstruction, rtol=0, atol=0)
    assert first.directions == 8
    assert set(first.raw_families) == {"B"}
    assert len(first.scout_pairs) == 1
    assert first.scout_pairs[0].family == "B"
    torch.testing.assert_close(
        partition.gather(first.basis, "A"),
        torch.zeros(partition.a_dimension, 8),
    )

    gradient = torch.linspace(-0.7, 0.8, partition.full_dimension)
    q_derivatives = first.basis.T @ gradient
    raw_derivatives = first.reconstruct_raw_derivatives(q_derivatives)
    raw_matrix = first.basis @ first.reconstruction
    torch.testing.assert_close(raw_derivatives, raw_matrix.T @ gradient)


def test_split_q8_basis_is_strictly_a_b_disjoint_with_two_scouts_each() -> None:
    partition = _partition()
    state = MatchedFocusState(partition)
    plan = build_focus_probe_plan(
        partition,
        state,
        torch.Generator().manual_seed(23),
        bootstrap_b_only=False,
    )
    assert plan.raw_families == ("A",) * 4 + ("B",) * 4
    assert plan.raw_kinds == ("fill", "fill", "scout", "scout") * 2
    assert [
        (pair.family, pair.first_raw_column, pair.second_raw_column) for pair in plan.scout_pairs
    ] == [
        ("A", 2, 3),
        ("B", 6, 7),
    ]
    torch.testing.assert_close(
        partition.gather(plan.basis[:, :4], "B"),
        torch.zeros(partition.b_dimension, 4),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        partition.gather(plan.basis[:, 4:], "A"),
        torch.zeros(partition.a_dimension, 4),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(plan.basis.T @ plan.basis, torch.eye(8), atol=2.0e-6, rtol=2.0e-6)


def test_half_batch_coordinate_api_builds_exact_raw_sketches_and_updates_only_planned_family() -> (
    None
):
    partition = _partition()
    state = MatchedFocusState(partition, FocusCovarianceConfig(ema_decay=0.5))
    plan = build_focus_probe_plan(
        partition,
        state,
        torch.Generator().manual_seed(31),
        bootstrap_b_only=True,
    )
    first_gradient = torch.linspace(-0.5, 0.6, partition.full_dimension)
    second_gradient = torch.linspace(0.8, -0.3, partition.full_dimension)
    first_coordinates = plan.basis.T @ first_gradient
    second_coordinates = plan.basis.T @ second_gradient
    pair = plan.scout_pairs[0]
    first_raw_direction = plan.raw_direction(pair.first_raw_column)
    second_raw_direction = plan.raw_direction(pair.second_raw_column)

    observations = state.update_from_half_batch_coordinates(
        plan,
        first_coordinates,
        second_coordinates,
    )
    assert len(observations) == 1
    observation = observations[0]
    assert observation.family == "B"
    expected_first_direction = partition.gather(first_raw_direction, "B")
    expected_second_direction = partition.gather(second_raw_direction, "B")
    expected_first_derivative = torch.dot(first_raw_direction, first_gradient)
    expected_second_derivative = torch.dot(second_raw_direction, second_gradient)
    torch.testing.assert_close(observation.first_derivative, expected_first_derivative)
    torch.testing.assert_close(observation.second_derivative, expected_second_derivative)
    torch.testing.assert_close(
        observation.first,
        expected_first_direction * expected_first_derivative,
    )
    torch.testing.assert_close(
        observation.second,
        expected_second_direction * expected_second_derivative,
    )
    assert state.family_state("A").update_count == 0
    assert state.family_state("B").update_count == 1
    assert state.persistent_numel <= state.persistent_numel_cap


def test_steady_plan_uses_learned_family_basis_then_preserves_q8() -> None:
    partition = _partition()
    state = MatchedFocusState(partition, FocusCovarianceConfig(ema_decay=0.0))
    for family in ("A", "B"):
        dimension = partition.family_dimension(family)
        first = torch.zeros(dimension)
        second = torch.zeros(dimension)
        first[0] = 2.0
        first[1] = 1.0
        second[0] = 1.0
        second[1] = 2.0
        state.family_state(family).update(first, second)
    plan = build_focus_probe_plan(
        partition,
        state,
        torch.Generator().manual_seed(41),
        bootstrap_b_only=False,
    )
    assert plan.directions == 8
    for family, offset in (("A", 0), ("B", 4)):
        learned = state.family_state(family).basis
        active_count = learned.shape[1]
        assert plan.raw_kinds[offset : offset + active_count] == ("active",) * active_count
        raw = plan.basis @ plan.reconstruction
        torch.testing.assert_close(
            partition.gather(raw[:, offset : offset + active_count], family),
            learned,
            atol=2.0e-6,
            rtol=2.0e-6,
        )


def test_matched_focus_state_serialization_round_trip_and_layout_guard() -> None:
    partition = _partition()
    state = MatchedFocusState(partition, FocusCovarianceConfig(ema_decay=0.8))
    plan = build_focus_probe_plan(
        partition,
        state,
        torch.Generator().manual_seed(53),
        bootstrap_b_only=False,
    )
    first = torch.linspace(-0.4, 0.3, 8)
    second = torch.linspace(0.2, 0.9, 8)
    state.update_from_half_batch_coordinates(plan, first, second)
    payload = state.state_dict()
    restored = MatchedFocusState.from_state_dict(partition, payload)
    assert restored.persistent_numel == state.persistent_numel
    for family in ("A", "B"):
        source = state.family_state(family)
        target = restored.family_state(family)
        assert target.update_count == source.update_count
        torch.testing.assert_close(target.basis, source.basis, rtol=0, atol=0)
        torch.testing.assert_close(target.eigenvalues, source.eigenvalues, rtol=0, atol=0)

    changed_layout = replace(
        _layout(),
        dtypes=("float16",) + ("float32",) * 3,
    )
    with pytest.raises(ValueError, match="different LoRA layout"):
        MatchedFocusState.from_state_dict(canonical_lora_partition(changed_layout), payload)


def test_focus_production_core_has_no_reverse_mode_api_calls_or_dense_covariance_state() -> None:
    tree = ast.parse(inspect.getsource(focus_module))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"backward", "grad"}:
            forbidden.append(node.func.attr)
        if isinstance(node.func, ast.Name) and node.func.id in {"backward", "grad"}:
            forbidden.append(node.func.id)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "autograd"
        ):
            forbidden.append("autograd")
    assert forbidden == []

    partition = _partition()
    state = MatchedFocusState(partition)
    assert state.persistent_numel == 0
    assert state.persistent_numel_cap == 2 * partition.full_dimension + 4
