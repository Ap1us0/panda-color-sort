"""Merge complete, consecutive headless evaluation batches into a new directory."""

import argparse
import csv
import json
from collections import Counter
from hashlib import sha256
from os.path import relpath
from pathlib import Path

from panda_sort_evaluate import ATTEMPT_FIELDS, FAILURE_FIELDS, OBJECT_FIELDS, SCENE_FIELDS, rates, write_report
from panda_sort_scenarios import MAX_EPISODE_STEPS


FIELDS = {"scenes": SCENE_FIELDS, "objects": OBJECT_FIELDS, "attempts": ATTEMPT_FIELDS, "failures": FAILURE_FIELDS}


def read_table(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"Unexpected columns in {path}")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Truncated or malformed rows in {path}")
    return rows


def read_batch(source: Path, max_steps: int) -> dict[str, list[dict[str, str]]]:
    tables = {name: read_table(source / f"{name}.csv", fields) for name, fields in FIELDS.items()}
    cases = {row["seed"]: row for row in tables["scenes"]}
    if not cases or len(cases) != len(tables["scenes"]):
        raise ValueError(f"Empty scene table or duplicate seed in {source}")
    for name in ("objects", "attempts", "failures"):
        for row in tables[name]:
            seed = row["seed"]
            if seed not in cases or ("cube_count" in row and row["cube_count"] != cases[seed]["cube_count"]):
                raise ValueError(f"Unmatched {name} record for seed {seed} in {source}")

    object_keys = {(row["seed"], row["cube"]) for row in tables["objects"]}
    attempt_keys = {(row["seed"], row["attempt"]) for row in tables["attempts"]}
    if len(object_keys) != len(tables["objects"]) or len(attempt_keys) != len(tables["attempts"]):
        raise ValueError(f"Duplicate object or attempt records in {source}")
    object_counts = Counter(row["seed"] for row in tables["objects"])
    placements = Counter(row["seed"] for row in tables["objects"] if csv_bool(row["correct"]))

    # Reconstruct the failure ledger and compare as a multiset, preserving repeat failures and full reasons.
    expected_failures = Counter()
    for seed, row in cases.items():
        count, correct, steps = (int(row[k]) for k in ("cube_count", "correct_cubes", "sim_steps"))
        passed = csv_bool(row["success"])
        if (
            count not in (2, 3) or correct != placements[seed] or not 0 <= correct <= count
            or not 0 <= steps <= max_steps or object_counts[seed] not in (0, count)
            or (passed and (correct != count or row["failure_type"] or row["failure_reason"]))
            or (not passed and not (row["failure_type"] and row["failure_reason"]))
            or (not object_counts[seed] and steps > 0)
        ):
            raise ValueError(f"Inconsistent scene or placement counts for seed {seed} in {source}")
        if not passed:
            expected_failures[(seed, row["cube_count"], "scene", "", "", row["failure_type"], row["failure_reason"])] += 1
    for row in tables["attempts"]:
        if not csv_bool(row["success"]):
            expected_failures[(row["seed"], cases[row["seed"]]["cube_count"], "attempt", row["cube"],
                               row["state"], row["failure_type"], row["reason"])] += 1
    if expected_failures != Counter(tuple(row[field] for field in FAILURE_FIELDS) for row in tables["failures"]):
        raise ValueError(f"Failure ledger differs from scene or attempt records in {source}")
    return tables


def csv_bool(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"Expected a CSV boolean, got {value!r}")
    return value == "True"


def merge_batches(
    sources: list[Path], output: Path, *, expected_scenes: int, max_steps: int = MAX_EPISODE_STEPS,
) -> list[dict]:
    combined: dict[str, list[dict[str, str]]] = {name: [] for name in FIELDS}
    manifest = []
    source_links = []
    for source in sources:
        tables = read_batch(source, max_steps)
        for name, rows in tables.items():
            combined[name].extend(rows)
        seeds = sorted(int(row["seed"]) for row in tables["scenes"])
        manifest.append({
            "directory": Path(relpath(source, output)).as_posix(), "scene_count": len(seeds),
            "first_seed": seeds[0], "last_seed": seeds[-1],
            "csv_sha256": {name: sha256((source / f"{name}.csv").read_bytes()).hexdigest() for name in FIELDS},
        })
        path = Path(relpath(source / "report.md", output)).as_posix()
        source_links.append(f"[{len(seeds)} 轮原始报告]({path})")

    scenes = sorted(combined["scenes"], key=lambda row: int(row["seed"]))
    if len(scenes) != expected_scenes:
        raise ValueError(f"Expected {expected_scenes} completed scenes, found {len(scenes)}")
    first = int(scenes[0]["seed"])
    for i, row in enumerate(scenes):
        if int(row["seed"]) != first + i or int(row["cube_count"]) != 2 + i % 2:
            raise ValueError(f"Duplicate, missing, or nonalternating seed near {row['seed']}")

    output.mkdir(parents=True, exist_ok=False)
    for name, fields in FIELDS.items():
        # Keep encounter order within a seed, including every failure and multi-line reason.
        rows = sorted(combined[name], key=lambda row: int(row["seed"]))
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_rows = [
        {**row, "seed": int(row["seed"]), "cube_count": int(row["cube_count"]),
         "success": csv_bool(row["success"]), "correct_cubes": int(row["correct_cubes"]),
         "sim_time_s": float(row["sim_time_s"]), "wall_time_s": float(row["wall_time_s"])}
        for row in scenes
    ]
    write_report(
        report_rows, output, max_steps=max_steps,
        run_summary=f"共合并 {len(scenes)} 个 headless 场景，保留所有成功和失败记录。"
                    f"来源为 {'、'.join(source_links)}，原始 CSV 校验值见 [来源清单](sources.json)。",
    )
    return report_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+", metavar="BATCH")
    parser.add_argument("--expected-scenes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.expected_scenes, args.max_steps) < 1:
        parser.error("expected-scenes and max-steps must be positive")
    try:
        rows = merge_batches(args.sources, args.output_dir, expected_scenes=args.expected_scenes, max_steps=args.max_steps)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    cube_rate, scene_rate, correct, cubes = rates(rows)
    print(f"{len(rows)} scenes | {correct}/{cubes} cubes ({cube_rate:.1%}) | "
          f"{sum(row['success'] for row in rows)}/{len(rows)} scenes ({scene_rate:.1%})")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
