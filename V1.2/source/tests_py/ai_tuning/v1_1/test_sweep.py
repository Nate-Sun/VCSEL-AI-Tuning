import csv
import json

import pytest

from vcsel_ai_tuning.v1_1.domain import DiscreteRange, RuntimeMode, TaskType
from vcsel_ai_tuning.v1_1.hardware_control import (
    HardwareTransactionError,
    RecoveryRequired,
)
from vcsel_ai_tuning.v1_1.repository import TaskStore
from vcsel_ai_tuning.v1_1.sweep import (
    PointEvidencePersistenceError,
    SweepHardware,
    SweepLimitExceeded,
    SweepMode,
    SweepPlan,
    SweepPoint,
    SweepPointStatus,
    SweepRunState,
    SweepRunner,
    create_bert_fir_sweep,
    create_i2c_sweep,
    export_sweep_artifacts,
    best_sweep_point,
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


def test_three_by_two_full_combination_produces_exactly_six_points():
    plan = create_i2c_sweep(
        ranges=(
            ("eq", DiscreteRange(1, 3, 1)),
            ("vref", DiscreteRange(4, 6, 2)),
        ),
        b0=B0_I2C,
        mode=SweepMode.FULL_COMBINATION,
        seed=7,
        max_points=6,
    )

    assert plan.point_count == 6
    assert len({tuple(sorted(point.requested.items())) for point in plan.points}) == 6


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
    def __init__(
        self,
        *,
        clock=None,
        persistence_s=0.0,
        fail_on_update=None,
    ):
        self.saved = []
        self.updated = []
        self.update_attempts = 0
        self.clock = clock
        self.persistence_s = persistence_s
        self.fail_on_update = fail_on_update

    def save_sweep_point(self, record):
        self.saved.append(record)
        if self.clock is not None:
            self.clock.advance(self.persistence_s)

    def update_sweep_point(self, record):
        self.update_attempts += 1
        if self.fail_on_update == self.update_attempts:
            raise OSError("injected durable update failure")
        self.updated.append(record)
        for index, existing in enumerate(self.saved):
            if existing.original_index == record.original_index:
                self.saved[index] = record
                return
        raise AssertionError("updated sweep point was never inserted")


class _StepClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class _TimedRestoreExecutor(FakeSweepExecutor):
    restore_after_each_point = True

    def __init__(self, clock, *, restore_fails=False, measurement_fails=False):
        super().__init__([{"tdecq": 2.5, "er": 3.6}])
        self.clock = clock
        self.restore_fails = restore_fails
        self.measurement_fails = measurement_fails
        self.final_readback = None

    def apply_point(self, point):
        result = dict(super().apply_point(point))
        self.clock.advance(0.2)
        result["write_readback_duration_s"] = 0.2
        return result

    def measure_point(self):
        self.clock.advance(2.0)
        self.clock.advance(3.5)
        if self.measurement_fails:
            error = RuntimeError("injected measurement failure")
            error.evidence = {
                "settle_duration_s": 2.0,
                "dca_duration_s": 3.5,
            }
            raise error
        result = dict(super().measure_point())
        result.update(settle_duration_s=2.0, dca_duration_s=3.5)
        return result

    def restore_b0(self):
        self.restore_b0_calls += 1
        self.clock.advance(0.4)
        if self.restore_fails:
            raise RuntimeError("injected restore failure")
        self.at_b0_value = True
        self.final_readback = dict(B0_I2C)

    def audit_evidence(self):
        return {"final_b0_readback": self.final_readback}


def _one_point_plan():
    return SweepPlan(
        hardware=SweepHardware.I2C_SWEEP,
        mode=SweepMode.FULL_COMBINATION,
        seed=5,
        max_points=1,
        points=(SweepPoint(0, 0, {"eq": 1, "vref": 7}),),
    )


def test_runner_persists_candidate_and_restore_round_timings():
    clock = _StepClock()
    executor = _TimedRestoreExecutor(clock)
    store = MemoryPointStore(clock=clock, persistence_s=0.1)

    result = SweepRunner(
        executor=executor,
        store=store,
        monotonic=clock,
    ).run(_one_point_plan())

    assert result.state is SweepRunState.COMPLETED
    assert executor.restore_b0_calls == 1
    assert len(store.saved) == 1
    # One update makes T6 timing durable before restore; the second appends T7.
    assert len(store.updated) == 2
    record = store.saved[0]
    assert record.write_readback_duration_s == pytest.approx(0.2)
    assert record.settle_duration_s == pytest.approx(2.0)
    assert record.dca_duration_s == pytest.approx(3.5)
    assert record.result_persistence_duration_s == pytest.approx(0.1)
    assert record.candidate_round_time_s == pytest.approx(5.8)
    assert record.restore_duration_s == pytest.approx(0.4)
    assert record.screening_round_with_restore_s == pytest.approx(6.2)
    assert record.restore_readback_result == B0_I2C
    assert record.restore_status == "VERIFIED"


def test_runner_keeps_failure_timings_when_restore_enters_recovery_required():
    clock = _StepClock()
    executor = _TimedRestoreExecutor(
        clock,
        restore_fails=True,
        measurement_fails=True,
    )
    store = MemoryPointStore(clock=clock, persistence_s=0.1)

    with pytest.raises(RecoveryRequired, match="restore"):
        SweepRunner(
            executor=executor,
            store=store,
            monotonic=clock,
        ).run(_one_point_plan())

    assert executor.restore_b0_calls == 1
    record = store.saved[0]
    assert record.status is SweepPointStatus.FAILED
    assert record.write_readback_duration_s == pytest.approx(0.2)
    assert record.settle_duration_s == pytest.approx(2.0)
    assert record.dca_duration_s == pytest.approx(3.5)
    assert record.result_persistence_duration_s == pytest.approx(0.1)
    assert record.candidate_round_time_s == pytest.approx(5.8)
    assert record.restore_duration_s == pytest.approx(0.4)
    assert record.screening_round_with_restore_s == pytest.approx(6.2)
    assert record.restore_readback_result is None
    assert record.restore_status == "RECOVERY_REQUIRED"


def test_runner_does_not_repeat_executor_verified_internal_rollback():
    clock = _StepClock()

    class _InternallyRestoredExecutor(_TimedRestoreExecutor):
        def apply_point(self, _point):
            self.clock.advance(0.3)
            self.at_b0_value = True
            self.final_readback = dict(B0_I2C)
            raise HardwareTransactionError(
                "candidate rejected and internally restored",
                rollback_verified=True,
            )

    executor = _InternallyRestoredExecutor(clock)
    store = MemoryPointStore(clock=clock, persistence_s=0.1)

    with pytest.raises(HardwareTransactionError, match="internally restored"):
        SweepRunner(
            executor=executor,
            store=store,
            monotonic=clock,
        ).run(_one_point_plan())

    assert executor.restore_b0_calls == 0
    assert store.saved[0].restore_status == "VERIFIED_INTERNAL_ROLLBACK"
    assert store.saved[0].candidate_round_time_s == pytest.approx(0.4)
    assert store.saved[0].screening_round_with_restore_s == pytest.approx(0.4)


def test_restore_evidence_persistence_failure_does_not_repeat_restore():
    clock = _StepClock()
    executor = _TimedRestoreExecutor(clock)
    store = MemoryPointStore(
        clock=clock,
        persistence_s=0.1,
        fail_on_update=2,
    )

    with pytest.raises(HardwareTransactionError) as caught:
        SweepRunner(
            executor=executor,
            store=store,
            monotonic=clock,
        ).run(_one_point_plan())

    assert caught.value.rollback_verified is True
    assert caught.value.evidence["restore_status"] == "VERIFIED"
    assert "evidence_persistence_error" in caught.value.evidence
    assert executor.restore_b0_calls == 1


def test_first_timing_update_failure_is_recovered_with_complete_point_evidence():
    clock = _StepClock()
    executor = _TimedRestoreExecutor(clock)
    store = MemoryPointStore(
        clock=clock,
        persistence_s=0.1,
        fail_on_update=1,
    )

    with pytest.raises(HardwareTransactionError) as caught:
        SweepRunner(
            executor=executor,
            store=store,
            monotonic=clock,
        ).run(_one_point_plan())

    assert isinstance(caught.value, PointEvidencePersistenceError)
    assert caught.value.rollback_verified is True
    assert executor.restore_b0_calls == 1
    assert store.update_attempts == 2
    record = store.saved[0]
    assert record.restore_status == "VERIFIED"
    assert record.result_persistence_duration_s == pytest.approx(0.1)
    assert record.candidate_round_time_s == pytest.approx(5.8)
    assert record.restore_duration_s == pytest.approx(0.4)
    assert record.screening_round_with_restore_s == pytest.approx(6.2)
    assert record.restore_readback_result == B0_I2C
    assert record.evidence_persistence_error == (
        "injected durable update failure"
    )
    assert caught.value.evidence == record.payload()


def test_timing_update_failure_does_not_repeat_executor_verified_rollback():
    clock = _StepClock()

    class _InternallyRestoredExecutor(_TimedRestoreExecutor):
        def apply_point(self, _point):
            self.clock.advance(0.3)
            self.at_b0_value = True
            self.final_readback = dict(B0_I2C)
            raise HardwareTransactionError(
                "candidate rejected and internally restored",
                rollback_verified=True,
            )

    executor = _InternallyRestoredExecutor(clock)
    store = MemoryPointStore(
        clock=clock,
        persistence_s=0.1,
        fail_on_update=1,
    )

    with pytest.raises(PointEvidencePersistenceError) as caught:
        SweepRunner(
            executor=executor,
            store=store,
            monotonic=clock,
        ).run(_one_point_plan())

    assert caught.value.rollback_verified is True
    assert executor.restore_b0_calls == 0
    assert store.update_attempts == 2
    assert store.saved[0].restore_status == "VERIFIED_INTERNAL_ROLLBACK"


def test_timing_update_failure_does_not_retry_failed_hardware_rollback():
    clock = _StepClock()

    class _RollbackFailedExecutor(_TimedRestoreExecutor):
        def apply_point(self, _point):
            self.clock.advance(0.3)
            self.final_readback = {"eq": 8, "vref": 7}
            raise RecoveryRequired(
                "candidate rollback failed",
                rollback_verified=False,
                evidence={
                    "last_observed_readback": dict(self.final_readback)
                },
            )

    executor = _RollbackFailedExecutor(clock)
    store = MemoryPointStore(
        clock=clock,
        persistence_s=0.1,
        fail_on_update=1,
    )

    with pytest.raises(RecoveryRequired) as caught:
        SweepRunner(
            executor=executor,
            store=store,
            monotonic=clock,
        ).run(_one_point_plan())

    assert type(caught.value).__name__ == "PointEvidenceRecoveryRequired"
    assert executor.restore_b0_calls == 0
    assert store.update_attempts == 2
    assert store.saved[0].restore_status == "RECOVERY_REQUIRED"
    assert store.saved[0].restore_readback_result == {"eq": 8, "vref": 7}


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
    assert len(store.saved) == 1
    assert store.saved[0].status is SweepPointStatus.FAILED
    assert store.saved[0].error_reason == "apply failed"


def test_best_point_uses_lowest_tdecq_then_highest_er():
    executor = FakeSweepExecutor(
        [
            {"tdecq": 2.4, "er": 3.0},
            {"tdecq": 2.2, "er": 2.8},
            {"tdecq": 2.2, "er": 3.2},
        ]
    )
    store = MemoryPointStore()
    SweepRunner(executor=executor, store=store).run(small_plan())

    best = best_sweep_point(store.saved)

    assert best is not None
    assert best.original_index == 2
    assert best.tdecq == 2.2
    assert best.er == 3.2


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
                "measurement_id": "measurement-0001",
                "acquisition_id": "acquisition-0001",
                "measurement_status": "SUCCESS",
                "dataset_status": "COMPLETE",
                "dataset_error": None,
                "dataset_bundle_path": "sweep-1/point-00000000",
                "raw": {
                    "point_id": "point-00000000",
                    "requested_settle_s": 0.1,
                    "actual_settle_s": 0.10012,
                    "application": {
                        "receipt": {
                            "apply_duration_s": 0.0019,
                            "readback_duration_s": 0.011,
                            "total_duration_s": 0.013,
                        }
                    },
                    "dca_raw_responses": {
                        "timing": {
                            "acquisition_duration_s": 0.0005,
                            "query_duration_s": 0.0074,
                            "wait_for_valid_duration_s": 8.078,
                            "total_duration_s": 8.086,
                        }
                    },
                },
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
    api_points = [record.payload() for record in store.saved]
    assert document["product_version"] == "1.1.0"
    assert document["campaign_type"] == "SWEEP"
    assert len(document["points"]) == 3
    assert document["best_point"]["tdecq"] == 2.3
    assert document["best_point"]["er"] == 3.8
    assert document["best_point"]["point_id"] == api_points[2]["point_id"]
    assert [point["point_id"] for point in document["points"]] == [
        point["point_id"] for point in api_points
    ]
    assert [row["point_id"] for row in rows] == [
        point["point_id"] for point in api_points
    ]
    assert rows[0]["point_id"] == "point-00000000"
    assert rows[0]["requested.eq"] == "1"
    assert rows[0]["tdecq"] == "2.5"
    assert rows[1]["status"] == "INVALID_TDECQ"
    assert "raw_response" in rows[0]
    csv_raw_response = json.loads(rows[0]["raw_response"])
    assert csv_raw_response == api_points[0]["raw_response"]
    assert document["points"][0]["raw_response"] == api_points[0]["raw_response"]
    assert document["points"][0]["measurement_id"] == "measurement-0001"
    assert document["points"][0]["acquisition_id"] == "acquisition-0001"
    assert document["points"][0]["measurement_status"] == "SUCCESS"
    assert document["points"][0]["dataset_status"] == "COMPLETE"
    assert rows[0]["dataset_bundle_path"] == "sweep-1/point-00000000"
    assert csv_raw_response["requested_settle_s"] == 0.1
    assert csv_raw_response["dca_raw_responses"]["timing"]["total_duration_s"] == 8.086
    timing_fields = {
        "write_readback_duration_s",
        "settle_duration_s",
        "dca_duration_s",
        "result_persistence_duration_s",
        "restore_duration_s",
        "candidate_round_time_s",
        "screening_round_with_restore_s",
        "restore_readback_result",
        "restore_status",
        "evidence_persistence_error",
    }
    assert timing_fields.issubset(document["points"][0])
    assert timing_fields.issubset(rows[0])
