import importlib.util
import sys
from pathlib import Path


def _load_repair_script_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "repair_learning_batch_quality_scores.py"
    spec = importlib.util.spec_from_file_location(
        "repair_learning_batch_quality_scores", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repair_script = _load_repair_script_module()


def test_build_repair_plan_backfills_nearest_unused_history():
    scanned_at = "2026-04-17T13:00:00+00:00"
    batches = [
        repair_script.BatchSnapshot(
            row_id=1,
            batch_id="b1",
            batch_name="batch one",
            group_id="g1",
            start_time=1000.0,
            end_time=1010.0,
            timestamp=1010.0,
            quality_score=None,
            error_message="",
            status="completed",
        ),
        repair_script.BatchSnapshot(
            row_id=2,
            batch_id="b2",
            batch_name="batch two",
            group_id="g1",
            start_time=1050.0,
            end_time=1060.0,
            timestamp=1060.0,
            quality_score=None,
            error_message="",
            status="completed",
        ),
    ]
    histories = [
        repair_script.HistorySnapshot(
            row_id=10,
            group_id="g1",
            timestamp=1008.0,
            quality_score=0.61,
            session_id="s1",
        ),
        repair_script.HistorySnapshot(
            row_id=11,
            group_id="g1",
            timestamp=1058.0,
            quality_score=0.73,
            session_id="s2",
        ),
    ]

    actions = repair_script.build_repair_plan(
        batches,
        histories,
        max_time_delta_seconds=120.0,
        scanned_at=scanned_at,
    )

    assert [action.action for action in actions] == ["backfill", "backfill"]
    assert actions[0].history_row_id == 10
    assert actions[1].history_row_id == 11
    assert actions[0].quality_score == 0.61
    assert actions[1].quality_score == 0.73


def test_build_repair_plan_marks_missing_without_zero_fallback():
    scanned_at = "2026-04-17T13:00:00+00:00"
    batches = [
        repair_script.BatchSnapshot(
            row_id=7,
            batch_id="b7",
            batch_name="legacy",
            group_id="g2",
            start_time=2000.0,
            end_time=2010.0,
            timestamp=2010.0,
            quality_score=None,
            error_message="existing message",
            status="completed",
        )
    ]
    histories = [
        repair_script.HistorySnapshot(
            row_id=21,
            group_id="g2",
            timestamp=9000.0,
            quality_score=0.88,
            session_id="late",
        )
    ]

    actions = repair_script.build_repair_plan(
        batches,
        histories,
        max_time_delta_seconds=60.0,
        scanned_at=scanned_at,
    )

    assert len(actions) == 1
    action = actions[0]
    assert action.action == "mark_missing"
    assert action.quality_score is None
    assert action.source == "historical_missing"
    assert "repair_status=historical_missing" in action.new_error_message
    assert "existing message" in action.new_error_message


def test_merge_error_message_removes_old_quality_marker_on_backfill():
    merged = repair_script.merge_error_message(
        "real error\n[quality_repair] repair_status=historical_missing",
        None,
    )

    assert merged == "real error"
