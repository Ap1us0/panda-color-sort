"""Sort two or three physical Panda cubes by their MuJoCo geom color."""

import argparse
import csv
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
from math import degrees, pi
from pathlib import Path

import mujoco
import numpy as np

from panda_pick_place import (
    DEFAULT_SEED, GOAL_RADIUS_M, GOAL_YAW_TOL_DEG, PICK_PLACE_SCENE, ROOT, Contacts, Gripper,
    State, Task, TrialResult, execute_pick, load_panda_spec, measure_contacts, planar_yaw,
    prepare_task, read_arm, reset_scene, square_yaw_offset, world_z_quaternion,
)


COLORS = {"red": (0.92, 0.09, 0.12, 1.0), "blue": (0.09, 0.35, 0.98, 1.0)}
# Separate starting areas keep the open fingers clear of the waiting cubes.
SPAWN_CENTERS = ((0.42, -0.193), (0.575, -0.193), (0.495, -0.037))
SPAWN_JITTER_M = 0.018
SLOTS = {
    "red": (("red_1", 0.41, 0.125), ("red_2", 0.445, 0.235)),
    "blue": (("blue_1", 0.605, 0.176),),
}


@dataclass(frozen=True)
class SortItem:
    name: str
    color: str
    slot: str
    task: Task


@dataclass(frozen=True)
class SortScene:
    model: mujoco.MjModel
    items: tuple[SortItem, ...]


@dataclass(frozen=True)
class CubeSpawn:
    name: str
    color: str
    x_m: float
    y_m: float
    yaw_rad: float


def prepare_sort(count: int, seed: int) -> SortScene:
    if count not in (2, 3):
        raise ValueError("Color sorting requires two or three cubes")
    rng = np.random.Generator(np.random.PCG64(seed))
    xy = np.array(SPAWN_CENTERS[:count]) + rng.uniform(-SPAWN_JITTER_M, SPAWN_JITTER_M, (count, 2))
    yaw = rng.uniform(-pi, pi, count)
    colors = rng.permutation(["red", "blue", "red"][:count])

    spawns = tuple(CubeSpawn(f"cube_{i}", str(color), *map(float, position), float(angle))
                   for i, (position, angle, color) in enumerate(zip(xy, yaw, colors), 1))
    return build_sort(spawns, prepare_task())


def build_sort(spawns: tuple[CubeSpawn, ...], base: Task, *, all_slots: bool = False) -> SortScene:
    source = base.model
    spec = load_panda_spec(PICK_PLACE_SCENE)
    spec.delete(spec.pair("cube_table"))
    spec.delete(spec.body("cube"))
    spec.delete(spec.site("place_goal"))
    world = spec.worldbody
    cube = source.geom("cube")
    cube_body = source.body("cube")
    pair = source.pair("cube_table")
    names = tuple(spawn.name for spawn in spawns)
    for spawn in spawns:
        name = spawn.name
        body = world.add_body(
            name=name, pos=[spawn.x_m, spawn.y_m, cube_body.pos[2]], quat=world_z_quaternion(spawn.yaw_rad),
            ipos=cube_body.ipos, mass=float(cube_body.mass[0]), inertia=cube_body.inertia, explicitinertial=True,
        )
        body.add_freejoint(name=f"{name}_free")
        body.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=cube.size, rgba=COLORS[spawn.color],
            condim=int(cube.condim[0]), friction=cube.friction, solref=cube.solref,
        )
        spec.add_pair(
            name=f"{name}_table", geomname1="table", geomname2=name,
            condim=int(pair.dim[0]), friction=pair.friction, solref=pair.solref, solimp=pair.solimp,
        )
    queues = {}
    marker_z = float(source.site("place_goal").pos[2])
    for color, slots in SLOTS.items():
        queues[color] = []
        used = len(slots) if all_slots else sum(spawn.color == color for spawn in spawns)
        for name, x, y in slots[:used]:
            world.add_site(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, marker_z],
                           size=[0.036, 0.036, 0.0005], rgba=COLORS[color])
            queues[color].append(name)

    model = spec.compile()
    shared = replace(base, model=model, arm=read_arm(model), gripper=Gripper.from_model(model),
                     table=model.geom("table").id)
    cubes = tuple(model.geom(name).id for name in names)
    items = []
    for name, cube_id in zip(names, cubes):
        # The simulated object color routes it to a matching unoccupied slot.
        red, _, blue, _ = model.geom_rgba[cube_id]
        color = "red" if red > blue else "blue"
        slot = queues[color].pop(0)
        goal = model.site(slot).pos.copy()
        goal[2] = shared.rest_z
        task = replace(shared, cube=cube_id, goal=goal, passive_cubes=tuple(g for g in cubes if g != cube_id))
        items.append(SortItem(name, color, slot, task))
    return SortScene(model, tuple(items))


def run_sort(
    scene: SortScene, observe: Callable[[SortItem, State, mujoco.MjData, Contacts], None] | None = None,
) -> tuple[mujoco.MjData, list[TrialResult]]:
    data = reset_scene(scene.model)
    results = []
    for item in scene.items:
        def observe_pick(state: State, live: mujoco.MjData, contacts: Contacts) -> None:
            if observe is not None:
                observe(item, state, live, contacts)

        result = execute_pick(item.task, data, observe_pick, align_yaw=True)
        results.append(result)
        if not result.success:
            break
    return data, results


def final_slot_check(item: SortItem, data: mujoco.MjData) -> tuple[float, float, bool]:
    task = item.task
    xyz = data.geom_xpos[task.cube]
    xy_error = float(np.linalg.norm(xyz[:2] - task.goal[:2]))
    yaw_error = abs(degrees(square_yaw_offset(planar_yaw(data.geom_xmat[task.cube]), 0.0)))
    contacts = measure_contacts(task, data, State.RETREAT)
    slide = int(task.model.joint(f"{item.name}_free").dofadr[0])
    motion = data.qvel[slide:slide + 6]
    ok = (
        xy_error <= GOAL_RADIUS_M and abs(xyz[2] - task.rest_z) <= 0.0015 and yaw_error <= GOAL_YAW_TOL_DEG
        and contacts.supported
        and np.linalg.norm(motion[:3]) < 0.005 and np.linalg.norm(motion[3:]) < 0.1
    )
    return xy_error, yaw_error, ok


def save_sort(scene: SortScene, seed: int, output: Path, *, video: bool) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    recording = nullcontext(None)
    if video:
        from panda_pick_place_video import record_video

        recording = record_video(scene.model, output / "color_sort.mp4", cube=scene.items[0].task.cube,
                                 fps=20, caption=f"{len(scene.items)} physical cubes  |  Red / blue slots")
    with (output / "tracking.csv").open("w", newline="", encoding="utf-8") as stream, recording as capture:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "cube", "color", "slot", "state", "cube_x_m", "cube_y_m", "cube_z_m",
                         "left_normal_n", "right_normal_n", "table_normal_n"))

        def observe(item: SortItem, state: State, data: mujoco.MjData, contacts: Contacts) -> None:
            if round(data.time / scene.model.opt.timestep) % 10 == 0:
                writer.writerow((f"{data.time:.3f}", item.name, item.color, item.slot, state.name,
                                 *data.geom_xpos[item.task.cube], contacts.left_normal_n,
                                 contacts.right_normal_n, contacts.table_normal_n))
            if capture is not None:
                capture(f"{item.color.upper()} {state.name}", data, contacts.left_normal_n, contacts.right_normal_n,
                        active_cube=item.task.cube)

        data, results = run_sort(scene, observe)

    correct = 0
    with (output / "sorting.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ("seed", "cube", "color", "slot", "goal_x_m", "goal_y_m", "final_slot_correct",
                   "episode_final_xy_error_m", "episode_final_yaw_error_deg", *(f.name for f in fields(TrialResult)))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i, item in enumerate(scene.items):
            result = results[i] if i < len(results) else TrialResult(state="SKIPPED", reason="earlier pick failed")
            xy_error, yaw_error, in_slot = final_slot_check(item, data)
            done = result.success and in_slot
            correct += int(done)
            writer.writerow({"seed": seed, "cube": item.name, "color": item.color, "slot": item.slot,
                             "goal_x_m": item.task.goal[0], "goal_y_m": item.task.goal[1],
                             "final_slot_correct": done, "episode_final_xy_error_m": xy_error,
                             "episode_final_yaw_error_deg": yaw_error, **asdict(result)})
            detail = f"PASS ({1000 * xy_error:.3f} mm from slot)" if done else (
                f"FAIL {result.state}: {result.reason or 'cube did not remain in its assigned slot'}"
            )
            print(f"{item.name}: {item.color} -> {item.slot} | {detail}", flush=True)
    print(f"Sorted {correct}/{len(scene.items)} cubes in {data.time:.2f} simulation seconds")
    return correct == len(scene.items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=(2, 3), default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    scene = prepare_sort(args.count, args.seed)
    output = args.output_dir or ROOT / "outputs" / f"color_sort_{args.count}"
    print(f"Sorting {args.count} cubes from seed {args.seed} | {output.resolve()}", flush=True)
    if not save_sort(scene, args.seed, output, video=args.video):
        parser.exit(1, "Color sorting did not complete\n")


if __name__ == "__main__":
    main()
