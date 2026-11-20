"""Generate or replay a seeded random color-sorting scene."""

import argparse
import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import lru_cache
from math import degrees, pi
from pathlib import Path

import mujoco
import numpy as np

from panda_color_sort import CubeSpawn, SLOTS, SortItem, SortScene, build_sort, final_slot_check
from panda_pick_place import (
    DEFAULT_SEED, PICK_PLACE_SCENE, ROOT, SPAWN_MAX_XY, SPAWN_MIN_XY, Contacts, State,
    Task, TrialResult, execute_pick, load_panda_spec, planar_yaw, plan_pickup, read_arm,
    reset_scene, square_yaw_offset, tcp_pose, tcp_position, world_z_quaternion,
)
from panda_cartesian import InverseKinematics, Trajectory


MIN_PAIR_DISTANCE_M = 0.135
MIN_SLOT_DISTANCE_M = 0.115
MAX_LAYOUT_DRAWS = 400
MAX_EPISODE_STEPS = 55_000
OBSTACLE_MARGIN_M = 0.002


class SceneGenerationError(ValueError):
    pass


@lru_cache(maxsize=1)
def base_task() -> Task:
    from panda_pick_place import prepare_task

    return prepare_task()


def generate_spawns(count: int, seed: int) -> tuple[CubeSpawn, ...]:
    if count not in (2, 3) or seed < 0:
        raise ValueError("Expected two or three cubes and a nonnegative seed")
    rng = np.random.Generator(np.random.PCG64(seed))
    base = base_task()
    model, home, arm = base.planning_model, base.planning_home, base.planning_arm
    ik = InverseKinematics(model, home, arm)
    _, home_quat = tcp_pose(home, arm)
    home_jaw = planar_yaw(home.xmat[arm.hand], axis=1)
    table = base.model.geom("table")
    cube_size = base.model.geom("cube").size
    slot_positions = np.array([slot[1:] for slots in SLOTS.values() for slot in slots])
    pairs = np.triu_indices(count, 1)
    for _ in range(MAX_LAYOUT_DRAWS):
        poses = rng.uniform((*SPAWN_MIN_XY, -pi), (*SPAWN_MAX_XY, pi), size=(count, 3))
        xy = poses[:, :2]
        if np.any(np.linalg.norm(xy[pairs[0]] - xy[pairs[1]], axis=1) < MIN_PAIR_DISTANCE_M):
            continue
        if np.any(np.linalg.norm(xy[:, None] - slot_positions, axis=2) < MIN_SLOT_DISTANCE_M):
            continue
        # The enclosing circle covers the cube's footprint at every yaw.
        if np.any(np.abs(xy - table.pos[:2]) + np.linalg.norm(cube_size[:2]) > table.size[:2]):
            continue
        try:
            for x, y, yaw in poses:
                quat = np.empty(4)
                mujoco.mju_mulQuat(quat, world_z_quaternion(square_yaw_offset(yaw, home_jaw)), home_quat)
                above = ik.solve(np.array([x, y, base.rest_z + 0.115]), home.qpos[arm.qpos], orientation=quat)
                ik.solve(np.array([x, y, base.rest_z]), above, orientation=quat)
        except (ValueError, RuntimeError):
            continue
        colors = rng.permutation(["red", "blue", "red"][:count])
        return tuple(CubeSpawn(f"cube_{i}", str(color), float(x), float(y), float(yaw))
                     for i, ((x, y, yaw), color) in enumerate(zip(poses, colors), 1))
    raise SceneGenerationError(f"no reachable layout after {MAX_LAYOUT_DRAWS} draws for seed {seed}")


def prepare_scenario(count: int, seed: int) -> tuple[SortScene, tuple[CubeSpawn, ...]]:
    spawns = generate_spawns(count, seed)
    scene = build_sort(spawns, base_task(), all_slots=True)
    data = reset_scene(scene.model)
    if any(contact.dist < -1e-6 for contact in data.contact):
        raise SceneGenerationError("compiled scene has a penetrating initial contact")
    return scene, spawns


def plan_around_cubes(task: Task, data: mujoco.MjData, pickup: np.ndarray, yaw_offset: float) -> Trajectory:
    spec = load_panda_spec(PICK_PLACE_SCENE)
    spec.delete(spec.pair("cube_table"))
    spec.delete(spec.body("cube"))
    for geom in task.passive_cubes:
        body_id = task.model.geom_bodyid[geom]
        spec.worldbody.add_geom(
            name=f"obstacle_{task.model.geom(geom).name}",
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=data.geom_xpos[geom], quat=data.xquat[body_id],
            size=task.model.geom_size[geom] + OBSTACLE_MARGIN_M,
        )
    model = spec.compile()
    planning = replace(task, planning_model=model, planning_home=reset_scene(model), planning_arm=read_arm(model))
    # Only arm joints are copied to the collision-planning model. Live cubes stay in dynamics.
    return plan_pickup(planning, pickup, yaw_offset=yaw_offset, start_angles=data.qpos[task.arm.qpos])


def classify_failure(state: str, reason: str) -> str:
    reason = reason.lower()
    if "budget" in reason or "timed out" in reason:
        return "timeout"
    if "collision" in reason or "unexpected contact" in reason or "penetration" in reason:
        return "collision"
    if "invalid dynamics" in reason or "joint angle or speed exceeded" in reason:
        return "dynamics_failure"
    if "unreachable" in reason and "did not converge" not in reason:
        return "unreachable"
    if state == "PLAN":
        return "trajectory_limits" if "velocity or acceleration" in reason else "ik_failure"
    if state == "GRASP":
        return "grasp_failure"
    if state in ("LIFT", "HOLD", "TRANSPORT") or "lost load-bearing" in reason:
        return "dropped"
    if state in ("PLACE", "RETREAT", "FINAL"):
        return "wrong_slot"
    return "dynamics_failure" if "dynamics" in reason else "settle_failure"


def ranked_assignments(scene: SortScene, data: mujoco.MjData, pending: dict[str, SortItem], occupied: dict[str, str]) -> list[SortItem]:
    if not pending:
        return []
    shared = scene.items[0].task
    tcp = tcp_position(data, shared.arm)
    options = []
    for item in pending.values():
        red, _, blue, _ = scene.model.geom_rgba[item.task.cube]
        color = "red" if red > blue else "blue"
        pickup = data.geom_xpos[item.task.cube]
        overhead = pickup + [0.0, 0.0, 0.115]
        for slot, _, _ in SLOTS[color]:
            goal = scene.model.site(slot).pos.copy()
            goal[2] = shared.rest_z
            if slot in occupied or any(
                other.name != item.name and np.linalg.norm(data.geom_xpos[other.task.cube, :2] - goal[:2]) < 0.070
                for other in scene.items
            ):
                continue
            cost = np.linalg.norm(tcp - overhead) + np.linalg.norm(pickup[:2] - goal[:2])
            options.append((cost, item.name, slot, replace(item, color=color, slot=slot, task=replace(item.task, goal=goal))))
    options.sort(key=lambda candidate: candidate[:3])
    return [option[-1] for option in options]


def run_scenario(
    scene: SortScene, seed: int, *, max_steps: int = MAX_EPISODE_STEPS,
    observe: Callable[[SortItem, State, mujoco.MjData, Contacts], None] | None = None,
) -> dict:
    data = reset_scene(scene.model)
    initial = {item.name: (*map(float, data.geom_xpos[item.task.cube]),
                           degrees(planar_yaw(data.geom_xmat[item.task.cube]))) for item in scene.items}
    pending = {item.name: item for item in scene.items}
    occupied = {}
    completed: dict[str, tuple[SortItem, TrialResult]] = {}
    attempts = []
    failure_type = failure_reason = ""

    for _ in scene.items:
        options = ranked_assignments(scene, data, pending, occupied)
        if not options:
            failure_type, failure_reason = "wrong_slot", "no unoccupied compatible slot"
            break
        for item in options:
            def observe_pick(state: State, live: mujoco.MjData, contacts: Contacts) -> None:
                if observe is not None:
                    observe(item, state, live, contacts)

            result = execute_pick(item.task, data, observe_pick if observe else None, align_yaw=True,
                                  max_episode_steps=max_steps, route_planner=plan_around_cubes)
            kind = "" if result.success else classify_failure(result.state, result.reason)
            attempts.append({
                "seed": seed, "attempt": len(attempts) + 1, "cube": item.name, "color": item.color, "slot": item.slot,
                "success": result.success, "state": result.state, "failure_type": kind, "reason": result.reason,
                "start_time_s": float(data.time - result.time_s), "end_time_s": float(data.time),
            })
            if result.success:
                occupied[item.slot] = item.name
                completed[item.name] = item, result
                del pending[item.name]
                break
            if result.state != "PLAN" or kind == "timeout":
                failure_type, failure_reason = kind, f"{item.name} / {result.state}: {result.reason}"
                break
        else:
            failure_type = attempts[-1]["failure_type"]
            failure_reason = "all candidate paths rejected; last error: " + attempts[-1]["reason"]
        if failure_type:
            break

    objects = []
    for original in scene.items:
        name = original.name
        x, y, z, yaw = initial[name]
        item, result = completed.get(name, (original, None))
        xy_error, yaw_error, in_slot = final_slot_check(item, data) if result is not None else (None, None, False)
        xyz = data.geom_xpos[original.task.cube]
        objects.append({
            "seed": seed, "cube_count": len(scene.items), "cube": name, "color": original.color,
            "initial_x_m": x, "initial_y_m": y, "initial_z_m": z, "initial_yaw_deg": yaw,
            "slot": item.slot if result is not None else "", "correct": bool(in_slot),
            "final_x_m": float(xyz[0]), "final_y_m": float(xyz[1]), "final_z_m": float(xyz[2]),
            "final_xy_error_m": xy_error, "final_yaw_error_deg": yaw_error,
            "hold_s": result.hold_s if result is not None else None,
        })
    correct = sum(obj["correct"] for obj in objects)
    success = correct == len(scene.items) and not failure_type
    if not success and not failure_type:
        failure_type, failure_reason = "wrong_slot", "one or more previously placed cubes no longer meet slot criteria"
    return {
        "scene": {
            "seed": seed, "cube_count": len(scene.items), "success": success, "correct_cubes": correct,
            "sim_time_s": float(data.time), "sim_steps": round(data.time / scene.model.opt.timestep),
            "grasp_order": " ".join(completed), "color_order": " ".join(item.color for item in scene.items),
            "failure_type": failure_type, "failure_reason": failure_reason,
        },
        "objects": objects, "attempts": attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=(2, 3), default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.seed < 0 or args.max_steps < 1:
        parser.error("seed must be nonnegative and max-steps must be positive")
    scene, spawns = prepare_scenario(args.count, args.seed)
    output = args.output_dir or ROOT / "outputs" / f"sort_replay_{args.seed}_{args.count}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "scenario.json").write_text(json.dumps({
        "seed": args.seed, "cube_count": args.count, "spawns": [asdict(spawn) for spawn in spawns],
    }, indent=2), encoding="utf-8")
    if args.manifest_only:
        return
    recording = nullcontext(None)
    if args.video:
        from panda_pick_place_video import record_video

        recording = record_video(scene.model, output / "sort.mp4", cube=scene.items[0].task.cube, fps=16,
                                 caption=f"Seed {args.seed}  |  {args.count} physical cubes")
    with recording as capture:
        def observe(item: SortItem, state: State, live: mujoco.MjData, contacts: Contacts) -> None:
            if capture is not None:
                capture(f"{item.color.upper()} {state.name}", live, contacts.left_normal_n, contacts.right_normal_n,
                        active_cube=item.task.cube)

        result = run_scenario(scene, args.seed, max_steps=args.max_steps, observe=observe if capture else None)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["scene"], indent=2), flush=True)
    if not result["scene"]["success"]:
        parser.exit(1, "Scenario did not complete; see result.json\n")


if __name__ == "__main__":
    main()
