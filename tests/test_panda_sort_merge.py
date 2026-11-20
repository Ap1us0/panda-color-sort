import csv
import json
from hashlib import sha256

import pytest

from panda_sort_evaluate import rates
from panda_sort_merge import FIELDS, merge_batches, read_batch, read_table


def write_table(directory, name, rows):
    with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS[name])
        writer.writeheader()
        writer.writerows(rows)


def make_batch(directory, seed, count, correct):
    directory.mkdir()
    passed = count == correct
    reason = "" if passed else "first line\nsecond line, kept verbatim"
    kind = "" if passed else "collision"
    write_table(directory, "scenes", [{
        "seed": seed, "cube_count": count, "success": passed, "correct_cubes": correct,
        "wall_time_s": 10, "sim_time_s": 20, "sim_steps": 10_000,
        "failure_type": kind, "failure_reason": reason,
    }])
    write_table(directory, "objects", [
        {"seed": seed, "cube_count": count, "cube": f"cube_{i + 1}", "correct": i < correct}
        for i in range(count)
    ])
    write_table(directory, "attempts", [] if passed else [{
        "seed": seed, "cube": "cube_2", "attempt": 2, "success": False, "state": "TRANSPORT",
        "failure_type": kind, "reason": reason,
    }])
    write_table(directory, "failures", [] if passed else [
        {"seed": seed, "cube_count": count, "scope": "attempt", "cube": "cube_2", "state": "TRANSPORT",
         "failure_type": kind, "reason": reason},
        {"seed": seed, "cube_count": count, "scope": "scene", "failure_type": kind, "reason": reason},
    ])


def test_merge_preserves_failures_and_recomputes_both_denominators(tmp_path):
    first, second, merged = [tmp_path / name for name in ("first", "second", "merged")]
    make_batch(first, 10, 2, 2)
    make_batch(second, 11, 3, 1)
    raw = (second / "failures.csv").read_bytes()

    rows = merge_batches([first, second], merged, expected_scenes=2)

    assert rates(rows) == (0.6, 0.5, 3, 5)
    assert (second / "failures.csv").read_bytes() == raw
    assert (merged / "failures.csv").read_bytes() == raw
    assert len(read_table(merged / "objects.csv", FIELDS["objects"])) == 5
    manifest = json.loads((merged / "sources.json").read_text(encoding="utf-8"))
    assert manifest[1]["csv_sha256"]["failures"] == sha256(raw).hexdigest()
    assert (merged / "metrics.png").is_file()


def test_merge_rejects_missing_failure_rows_or_duplicate_seeds(tmp_path):
    source = tmp_path / "source"
    make_batch(source, 10, 2, 1)
    with pytest.raises(ValueError, match="Duplicate, missing, or nonalternating"):
        merge_batches([source, source], tmp_path / "merged", expected_scenes=2)
    write_table(source, "failures", [])
    with pytest.raises(ValueError, match="Failure ledger"):
        read_batch(source, 55_000)
