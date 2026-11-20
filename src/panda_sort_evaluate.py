"""Run durable headless evaluations of reproducible color-sorting scenes."""

import argparse
import csv
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from time import perf_counter

from panda_pick_place import DEFAULT_SEED, ROOT
from panda_sort_scenarios import MAX_EPISODE_STEPS, SceneGenerationError, prepare_scenario, run_scenario


SCENE_FIELDS = (
    "seed", "cube_count", "success", "correct_cubes", "wall_time_s", "sim_time_s", "sim_steps",
    "color_order", "grasp_order", "failure_type", "failure_reason",
)
OBJECT_FIELDS = (
    "seed", "cube_count", "cube", "color", "initial_x_m", "initial_y_m", "initial_z_m", "initial_yaw_deg",
    "slot", "correct", "final_x_m", "final_y_m", "final_z_m", "final_xy_error_m", "final_yaw_error_deg", "hold_s",
)
ATTEMPT_FIELDS = (
    "seed", "attempt", "cube", "color", "slot", "success", "state", "failure_type", "reason",
    "start_time_s", "end_time_s",
)
FAILURE_FIELDS = ("seed", "cube_count", "scope", "cube", "state", "failure_type", "reason")


def evaluate_case(seed: int, count: int, max_steps: int) -> dict:
    tick = perf_counter()
    try:
        scene, _ = prepare_scenario(count, seed)
        result = run_scenario(scene, seed, max_steps=max_steps)
    except Exception as error:
        kind = "unreachable" if isinstance(error, SceneGenerationError) else "program_error"
        result = {
            "scene": {
                "seed": seed, "cube_count": count, "success": False, "correct_cubes": 0,
                "sim_time_s": 0.0, "sim_steps": 0, "failure_type": kind,
                "failure_reason": str(error) if kind == "unreachable" else traceback.format_exc(),
            },
            "objects": [], "attempts": [],
        }
    result["scene"]["wall_time_s"] = perf_counter() - tick
    return result


def make_cases(count: int, seed: int) -> list[tuple[int, int]]:
    return [(seed + i, 2 + i % 2) for i in range(count)]


def run_cases(cases: list[tuple[int, int]], output: Path, *, workers: int, max_steps: int) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    with ExitStack() as stack:
        writers = {}
        streams = []
        for name, columns in (
            ("scenes", SCENE_FIELDS), ("objects", OBJECT_FIELDS), ("attempts", ATTEMPT_FIELDS),
            ("failures", FAILURE_FIELDS),
        ):
            stream = stack.enter_context((output / f"{name}.csv").open("w", encoding="utf-8", newline=""))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            stream.flush()
            streams.append(stream)
            writers[name] = writer
        executor = stack.enter_context(ProcessPoolExecutor(max_workers=workers))
        futures = {executor.submit(evaluate_case, seed, count, max_steps): (seed, count) for seed, count in cases}
        for future in as_completed(futures):
            seed, count = futures[future]
            try:
                result = future.result()
            except Exception:
                result = {
                    "scene": {"seed": seed, "cube_count": count, "success": False, "correct_cubes": 0,
                              "wall_time_s": 0.0, "sim_time_s": 0.0, "sim_steps": 0,
                              "failure_type": "program_error", "failure_reason": traceback.format_exc()},
                    "objects": [], "attempts": [],
                }
            row = result["scene"]
            completed.append(row)
            writers["scenes"].writerow(row)
            writers["objects"].writerows(result["objects"])
            writers["attempts"].writerows(result["attempts"])
            for attempt in result["attempts"]:
                if not attempt["success"]:
                    writers["failures"].writerow({"seed": seed, "cube_count": count, "scope": "attempt",
                        **{key: attempt[key] for key in ("cube", "state", "failure_type", "reason")}})
            if not row["success"]:
                writers["failures"].writerow({"seed": seed, "cube_count": count, "scope": "scene",
                    "failure_type": row["failure_type"], "reason": row["failure_reason"]})
            for stream in streams:
                stream.flush()
            print(f"[{len(completed):02}/{len(cases)}] seed {seed} | {row['correct_cubes']}/{count} cubes | "
                  f"{'PASS' if row['success'] else row['failure_type']} | {row['wall_time_s']:.1f} s wall", flush=True)
    return sorted(completed, key=lambda row: row["seed"])


def rates(rows: list[dict]) -> tuple[float, float, int, int]:
    correct = sum(row["correct_cubes"] for row in rows)
    cubes = sum(row["cube_count"] for row in rows)
    complete = sum(row["success"] for row in rows)
    return correct / cubes if cubes else 0, complete / len(rows) if rows else 0, correct, cubes


def write_report(rows: list[dict], output: Path, *, max_steps: int, run_summary: str) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    groups = {"全部": rows, "2 块": [r for r in rows if r["cube_count"] == 2],
              "3 块": [r for r in rows if r["cube_count"] == 3]}
    report = [
        "# 随机红蓝分拣实验", "",
        run_summary,
        f"种子范围 {rows[0]['seed']}–{rows[-1]['seed']}，每轮最多 {max_steps:,} 个物理步。",
        "整轮成功要求所有方块在最终复查时仍稳定留在正确槽位。未完成的方块计入分母。", "",
        "| 组别 | 正确落槽方块 | 单块正确率 | 完整场景 | 整轮成功率 | 成功轮仿真耗时中位数 | 单轮墙钟耗时中位数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, values in groups.items():
        if not values:
            continue
        obj_rate, scene_rate, correct, count = rates(values)
        passed = [r["sim_time_s"] for r in values if r["success"]]
        sim_med = f"{median(passed):.2f} s" if passed else "—"
        wall_med = median(row["wall_time_s"] for row in values)
        report.append(f"| {label} | {correct}/{count} | {obj_rate:.1%} | "
                      f"{sum(row['success'] for row in values)}/{len(values)} | {scene_rate:.1%} | {sim_med} | {wall_med:.2f} s |")
    counts = Counter(row["failure_type"] for row in rows if not row["success"])
    report.extend(["", "## 失败与复现", ""])
    if counts:
        for kind, count in sorted(counts.items()):
            report.append(f"- `{kind}` 共 {count} 轮。")
        report.extend(["", "| seed | 方块数 | 类型 | 原因 |", "| ---: | ---: | --- | --- |"])
        for row in rows:
            if not row["success"]:
                reason = row["failure_reason"].splitlines()[-1].replace("|", "\\|")
                report.append(f"| {row['seed']} | {row['cube_count']} | `{row['failure_type']}` | {reason} |")
    else:
        report.append("本批次无整轮失败。被拒绝后改选的路径记录在尝试与失败 CSV 中。")
    report.extend([
        "", "每个 seed 可用 `panda_sort_scenarios.py --seed <seed> --count <count> --max-steps <steps>` 独立复现。",
        "", "[场景 CSV](scenes.csv) · [方块 CSV](objects.csv) · [尝试 CSV](attempts.csv) · [失败 CSV](failures.csv) · [统计图](metrics.png)",
        "",
    ])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")

    figure = Figure(figsize=(11.2, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle(f"{len(rows)} evaluated color-sort scenes")
    success_ax, time_ax = figure.subplots(1, 2)
    categories = ["All", "Two cubes", "Three cubes"]
    data = list(groups.values())
    x = list(range(3))
    success_ax.bar([i - 0.17 for i in x], [rates(g)[0] * 100 for g in data], 0.34, label="Per cube", color="#157b78")
    success_ax.bar([i + 0.17 for i in x], [rates(g)[1] * 100 for g in data], 0.34, label="Per scene", color="#4d67aa")
    success_ax.set(xticks=x, xticklabels=categories, ylim=(0, 108), ylabel="Success (%)", title="Correct placement rates")
    success_ax.axhline(80, color="#bb6442", linestyle="--", linewidth=1)
    success_ax.legend(loc="lower right")
    for i, group in enumerate(data[1:], 1):
        if group:
            time_ax.scatter([row["sim_time_s"] for row in group], [row["wall_time_s"] for row in group],
                            label=categories[i], s=30, alpha=0.65)
    time_ax.set(xlabel="Simulation time (s)", ylabel="Per-scene wall time (s)", title="Timing by object count")
    time_ax.legend()
    for ax in (success_ax, time_ax):
        ax.grid(axis="y", alpha=0.22)
    figure.savefig(output / "metrics.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke-count", type=int, default=5)
    parser.add_argument("--smoke-seed", type=int, default=7300)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "robust_sort")
    args = parser.parse_args()
    if min(args.scenes, args.smoke_count, args.seed, args.smoke_seed) < 0 or min(args.jobs, args.max_steps) < 1:
        parser.error("counts and seeds must be nonnegative; jobs and max-steps must be positive")
    if not (args.scenes or args.smoke_count):
        parser.error("request at least one scenario")
    for label, count, seed in (("smoke", args.smoke_count, args.smoke_seed), ("evaluation", args.scenes, args.seed)):
        if not count:
            continue
        output = args.output_dir / label
        tick = perf_counter()
        rows = run_cases(make_cases(count, seed), output, workers=min(count, args.jobs), max_steps=args.max_steps)
        elapsed = perf_counter() - tick
        write_report(
            rows, output, max_steps=args.max_steps,
            run_summary=f"共完成 {len(rows)} 个场景，使用 {min(count, args.jobs)} 个 headless 工作进程，批次耗时 {elapsed:.2f} s。",
        )
        objects, scenes, _, _ = rates(rows)
        print(f"{label}: {objects:.1%} per cube, {scenes:.1%} per scene | {output.resolve()}", flush=True)
        if label == "smoke" and any(row["failure_type"] == "program_error" for row in rows):
            parser.exit(1, "Smoke test found a program error; see failures.csv\n")


if __name__ == "__main__":
    main()
