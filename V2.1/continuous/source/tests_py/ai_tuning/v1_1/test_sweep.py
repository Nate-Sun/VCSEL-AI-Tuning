import csv
import json

import pytest

from vcsel_ai_tuning.v1_1.domain import DiscreteRange, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.repository import TaskStore
from vcsel_ai_tuning.v1_1.sweep import (
    SweepHardware,
    SweepLimitExceeded,
    SweepMode,
    SweepPointStatus,
    SweepRunState,
    SweepRunner,
    create_bert_fir_sweep,
    create_i2c_sweep,
    export_sweep_artifacts,
)


B0_I2C = {"eq": 20, "vref": 7}
B0_FIR = (0, 9, -28, 104, -8, 1, -3)


def test_i2c_single_parameter_changes_one_field_and_keeps_others_at_b0():
    plan = create_i2c_sweep(
        ranges=(
            ("eq", DiscreteRange(1, 5, 2)),
            ("vref", DiscreteRange(2, 4, 2)),
        ),
        b0=B0_I2C,
        mode=SweepMode.SINGLE_PARAMETER,
        seed=17,
        max_points=5,
    )

    assert plan.hardware is SweepHardware.I2C_SWEEP
    assert [point.requested for point in plan.points] == [
        {"eq": 1, "vref": 7},
        {"eq": 3, "vref": 7},
        {"eq": 5, "vref": 7},
        {"eq": 20, "vref": 2},
        {"eq": 20, "vref": 4},
    ]
    assert [point.group_end for point in plan.points] == [
        False,
        False,
        True,
        False,
        True,
    ]


def test_full_combination_is_deterministic_preserves_both_indexes_and_never_samples():
    arguments = dict(
        ranges=(
            ("eq", DiscreteRange(1, 2, 1)),
            ("vref", DiscreteRange(4, 6, 2)),
        ),
        b0=B0_I2C,
        mode=SweepMode.FULL_COMBINATION,
        seed=91,
        max_points=4,
    )

    first = create_i2c_sweep(**arguments)
    second = create_i2c_sweep(**arguments)

    assert len(first.points) == 4
    assert [point.original_index for point in first.points] == [
        point.original_index for point in second.points
    ]
    assert sorted(point.original_index for point in first.points) == [0, 1, 2, 3]
    assert [point.execution_index for point in first.points] == [0, 1, 2, 3]
    assert [point.requested for point in first.points] != [
        {"eq": 1, "vref": 4},
        {"eq": 1, "vref": 6},
        {"eq": 2, "vref": 4},
        {"eq": 2, "vref": 6},
    ]


def test_full_combination_over_max_points_fails_instead_of_sampling():
    with pytest.raises(SweepLimitExceeded) as caught:
        create_i2c_sweep(
            ranges=(
                ("eq", DiscreteRange(1, 3, 1)),
                ("vref", DiscreteRange(4, 6, 1)),
            ),
            b0=B0_I2C,
            mode=SweepMode.FULL_COMBINATION,
            seed=0,
            max_points=8,
        )

    assert caught.value.point_count == 9
    assert caught.value.max_points == 8


def test_bert_sweep_marks_illegal_combinations_without_hardware_execution():
    plan = create_bert_fir_sweep(
        ranges=(
            ("main", DiscreteRange(0, 100, 100)),
            ("pre1", DiscreteRange(-80, 0, 80)),
        ),
        b0=B0_FIR,
        mode=SweepMode.FULL_COMBINATION,
        seed=3,
        max_points=4,
    )

    assert plan.hardware is SweepHardware.BERT_FIR_SWEEP
    skipped = [
        point for point in plan.points
        if point.status is SweepPointStatus.SKIPPED_CONSTRAINT
    ]
    assert skipped
    assert all(point.constraint_reason for point in skipped)
    assert any("Main" in point.constraint_reason for point in skipped)
    assert any("L1" in point.constraint_reason for point in skipped)


class FakeSweepExecutor:
    def __init__(self, measurements, *, at_b0=True, apply_error_at=None):
        self.measurements = list(measurements)
        self.at_b0_value = at_b0
        self.apply_error_at = apply_error_at
        self.applied = []
        self.restored_parameters = []
        self.restore_b0_calls = 0
        self.preflight_calls = 0

    def preflight(self):
        self.preflight_calls += 1

    def is_at_b0(self):
        return self.at_b0_value

    def restore_b0(self):
        self.restore_b0_calls += 1
        self.at_b0_value = True

    def restore_parameter(self, name):
        self.restored_parameters.append(name)

    def apply_point(self, point):
        if point.original_index == self.apply_error_at:
            raise RuntimeError("apply failed")
        self.applied.append(point.original_index)
        self.at_b0_value = False
        return {
            "requested": dict(point.requested),
            "applied": dict(point.requested),
            "readback": dict(point.requested),
        }

    def measure_point(self):
        return self.measurements.pop(0)


class MemoryPointStore:
    def __init__(self):
        self.saved = []

    def save_sweep_point(self, record):
        self.saved.append(record)


def small_plan():
    return create_i2c_sweep(
        ranges=(("eq", DiscreteRange(1, 3, 1)),),
        b0=B0_I2C,
        mode=SweepMode.SINGLE_PARAMETER,
        seed=5,
        max_points=3,
    )


def test_runner_persists_every_point_and_invalid_tdecq_continues():
    executor = FakeSweepExecutor(
        [
            {"tdecq": 2.5, "er": 3.6, "raw": "one"},
            {"tdecq": None, "er": 3.7, "raw": "invalid"},
            {"tdecq": 2.3, "er": 3.8, "raw": "three"},
        ]
    )
    store = MemoryPointStore()
    runner = SweepRunner(executor=executor, store=store)

    result = runner.run(small_plan())

    assert result.state is SweepRunState.COMPLETED
    assert executor.applied == [0, 1, 2]
    assert [record.status for record in store.saved] == [
        SweepPointStatus.COMPLETED,
        SweepPointStatus.INVALID_TDECQ,
        SweepPointStatus.COMPLETED,
    ]
    assert executor.restored_parameters == ["eq"]
    assert executor.restore_b0_calls == 1


def test_runner_never_applies_constraint_skipped_point_but_persists_it():
    plan = create_bert_fir_sweep(
        ranges=(("main", DiscreteRange(0, 1, 1)),),
        b0=B0_FIR,
        mode=SweepMode.SINGLE_PARAMETER,
        seed=1,
        max_points=2,
    )
    executor = FakeSweepExecutor([{"tdecq": 2.4, "er": 3.7}])
    store = MemoryPointStore()

    SweepRunner(executor=executor, store=store).run(plan)

    assert len(executor.applied) == 1
    assert [record.status for record in store.saved] == [
        SweepPointStatus.SKIPPED_CONSTRAINT,
        SweepPointStatus.COMPLETED,
    ]


def test_pause_restores_b0_and_resume_skips_persisted_points_in_original_order():
    plan = small_plan()
    first_executor = FakeSweepExecutor(
        [{"tdecq": 2.5, "er": 3.6}]
    )
    first_store = MemoryPointStore()
    checks = iter((False, True))
    first = SweepRunner(executor=first_executor, store=first_store).run(
        plan,
        pause_requested=lambda: next(checks),
    )

    assert first.state is SweepRunState.PAUSED
    assert first.completed_original_indexes == (0,)
    assert first_executor.restore_b0_calls == 1

    resumed_executor = FakeSweepExecutor(
        [
            {"tdecq": 2.4, "er": 3.7},
            {"tdecq": 2.3, "er": 3.8},
        ],
        at_b0=False,
    )
    resumed_store = MemoryPointStore()
    resumed = SweepRunner(
        executor=resumed_executor, store=resumed_store
    ).run(plan, completed_original_indexes=(0,))

    assert resumed.state is SweepRunState.COMPLETED
    assert resumed_executor.preflight_calls == 1
    assert resumed_executor.restore_b0_calls == 2  # resume guard + completion
    assert resumed_executor.applied == [1, 2]


def test_apply_failure_restores_b0_and_propagates():
    executor = FakeSweepExecutor([], apply_error_at=0)
    store = MemoryPointStore()

    with pytest.raises(RuntimeError, match="apply failed"):
        SweepRunner(executor=executor, store=store).run(small_plan())

    assert executor.restore_b0_calls == 1


def test_sqlite_sweep_points_are_immediately_persisted_and_paginated(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    with TaskStore(database) as store:
        store.create_task(
            task_id="sweep-1",
            task_type=TaskType.SWEEP,
            runtime_mode=RuntimeMode.FAKE,
            config_fingerprint="c" * 64,
            payload={"hardware": "I2C_SWEEP"},
        )
        runner_store = store.sweep_point_sink("sweep-1")
        executor = FakeSweepExecutor(
            [
                {"tdecq": 2.5, "er": 3.6},
                {"tdecq": None, "er": 3.7},
                {"tdecq": 2.3, "er": 3.8},
            ]
        )
        SweepRunner(executor=executor, store=runner_store).run(small_plan())

        page = store.sweep_points("sweep-1", offset=1, limit=1)

    assert page.total == 3
    assert len(page.items) == 1
    assert page.items[0].original_index == 1
    assert page.items[0].status is SweepPointStatus.INVALID_TDECQ


def test_json_and_flat_csv_exports_include_audit_fields(tmp_path):
    executor = FakeSweepExecutor(
        [
            {
                "tdecq": 2.5,
                "er": 3.6,
                "raw": {"status": "CORR"},
            },
            {"tdecq": None, "er": 3.7, "raw": {"status": "INV"}},
            {"tdecq": 2.3, "er": 3.8, "raw": {"status": "CORR"}},
        ]
    )
    store = MemoryPointStore()
    SweepRunner(executor=executor, store=store).run(small_plan())

    paths = export_sweep_artifacts(
        directory=tmp_path,
        task_id="sweep-1",
        plan=small_plan(),
        records=tuple(store.saved),
        product_version="1.1.0",
        schema_version=1,
        runtime_mode="FAKE",
        b0_hash="b" * 64,
        configuration_fingerprint="c" * 64,
    )

    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    with paths.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert document["product_version"] == "1.1.0"
    assert document["campaign_type"] == "SWEEP"
    assert len(document["points"]) == 3
    assert rows[0]["requested.eq"] == "1"
    assert rows[0]["tdecq"] == "2.5"
    assert rows[1]["status"] == "INVALID_TDECQ"
    assert "raw_response" in rows[0]
