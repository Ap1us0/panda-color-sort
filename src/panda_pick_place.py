"""Run fixed or seeded random-pose Panda pick-and-place with physical contacts."""

import argparse
import csv
from math import atan2, cos, degrees, pi, sin
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum, auto
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np

from panda_cartesian import (
    MAX_TRACKING_SPEED, Arm, PoseWaypoint, Trajectory, joint_controls, plan_trajectory, read_arm,
    tcp_pose, tcp_position,
)
from panda_position import ROOT, load_panda_spec


PICK_PLACE_SCENE = ROOT / "models" / "franka_emika_panda" / "pick_place.xml"
MIN_LIFT_M = 0.05
MIN_HOLD_S = 1.0
MIN_NORMAL_N = 0.15
GOAL_RADIUS_M = 0.015
GOAL_YAW_TOL_DEG = 5.0
MAX_CONTACT_DEPTH_M = 0.001
MIN_STABLE_S = 1.0
SPAWN_MIN_XY = (0.40, -0.22)
SPAWN_MAX_XY = (0.60, 0.04)
DEFAULT_SEED = 20261120


@dataclass(frozen=True)
class Gripper:
    actuator: int
    fingers: tuple[int, int]
    bodies: tuple[int, int]
    closed: float
    opened: float

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "Gripper":
        actuator = model.actuator("actuator8")
        tendon = model.tendon("split")
        left = model.joint("finger_joint1")
        right = model.joint("finger_joint2")
        fingers = left.id, right.id
        start = model.tendon_adr[tendon.id]
        stop = start + model.tendon_num[tendon.id]
        if not (
            actuator.trntype[0] == mujoco.mjtTrn.mjTRN_TENDON
            and actuator.trnid[0] == tendon.id
            and actuator.ctrllimited[0] and actuator.forcelimited[0]
            and all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE for j in fingers)
            and np.array_equal(model.wrap_objid[start:stop], fingers)
            and np.allclose(model.wrap_prm[start:stop], 0.5)
        ):
            raise ValueError("Expected actuator8 to drive both sliding fingers through the split tendon")
        return cls(
            actuator.id, (int(left.qposadr[0]), int(right.qposadr[0])),
            (model.body("left_finger").id, model.body("right_finger").id),
            *map(float, actuator.ctrlrange),
        )

    def command(self, data: mujoco.MjData, opening: float) -> None:
        """Set normalized opening, where 0 closes, 1 opens, and a constant holds."""
        if not 0.0 <= opening <= 1.0:
            raise ValueError("Gripper opening must be within [0, 1]")
        data.ctrl[self.actuator] = self.closed + opening * (self.opened - self.closed)


def load_scene(*, with_cube: bool = True) -> mujoco.MjModel:
    spec = load_panda_spec(PICK_PLACE_SCENE)
    if not with_cube:
        spec.delete(spec.pair("cube_table"))
        spec.delete(spec.body("cube"))
    return spec.compile()


def reset_scene(
    model: mujoco.MjModel, cube_xy: np.ndarray | None = None, *, cube_yaw: float | None = None,
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    if cube_xy is not None or cube_yaw is not None:
        start = int(model.joint("cube_free").qposadr[0])
        xy = data.qpos[start:start + 2] if cube_xy is None else np.asarray(cube_xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Cube spawn must be a finite world XY position")
        quaternion = data.qpos[start + 3:start + 7]
        if cube_yaw is not None:
            if not np.isfinite(cube_yaw):
                raise ValueError("Cube yaw must be finite")
            quaternion = world_z_quaternion(cube_yaw)
        table, cube = model.geom("table"), model.geom("cube")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        extent = np.abs(rotation.reshape(3, 3)[:2]) @ cube.size
        if np.any(np.abs(xy - table.pos[:2]) + extent > table.size[:2]):
            raise ValueError("Cube footprint must fit on the tabletop")
        # Cube state is set only here, before the first dynamics step.
        data.qpos[start:start + 2] = xy
        if cube_yaw is not None:
            data.qpos[start + 3:start + 7] = quaternion
    mujoco.mj_forward(model, data)
    return data


def world_z_quaternion(yaw: float) -> np.ndarray:
    return np.array([cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)])


def planar_yaw(matrix: np.ndarray, axis: int = 0) -> float:
    """World heading of the selected local axis of a row-major rotation matrix."""
    return atan2(matrix[3 + axis], matrix[axis])


def square_yaw_offset(cube_yaw: float, jaw_yaw: float) -> float:
    """Shortest signed rotation from the jaw axis to any cube face normal."""
    return (cube_yaw - jaw_yaw + pi / 4) % (pi / 2) - pi / 4


def validation_pickups() -> np.ndarray:
    x0, y0 = SPAWN_MIN_XY
    x1, y1 = SPAWN_MAX_XY
    return np.array([(0.5 * (x0 + x1), 0.5 * (y0 + y1)), (x0, y0), (x0, y1), (x1, y0), (x1, y1)])


def sample_pickups(count: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.uniform(SPAWN_MIN_XY, SPAWN_MAX_XY, size=(count, 2))


def validation_poses() -> np.ndarray:
    return np.column_stack((validation_pickups(), np.deg2rad([-136.0, -75.0, -46.0, 45.0, 160.0])))


def sample_pickup_poses(count: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.uniform((*SPAWN_MIN_XY, -pi), (*SPAWN_MAX_XY, pi), size=(count, 3))


class State(Enum):
    SETTLE = auto()
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    HOLD = auto()
    TRANSPORT = auto()
    PLACE = auto()
    RETREAT = auto()


@dataclass(frozen=True)
class Task:
    model: mujoco.MjModel
    arm: Arm
    gripper: Gripper
    cube: int
    table: int
    rest_z: float
    goal: np.ndarray
    planning_model: mujoco.MjModel
    planning_home: mujoco.MjData
    planning_arm: Arm
    passive_cubes: tuple[int, ...] = ()


def prepare_task() -> Task:
    model = load_scene()
    arm = read_arm(model)
    gripper = Gripper.from_model(model)
    table, cube = model.geom("table").id, model.geom("cube").id
    height = model.geom_pos[table, 2] + model.geom_size[table, 2] + model.geom_size[cube, 2]
    goal = model.site("place_goal").pos.copy()
    goal[2] = height

    # Plan with the same robot and table. A free body frozen at its starting
    # position is not a valid model of a carried object during IK preflight.
    # The live model includes the cube for every physical step.
    planning_model = load_scene(with_cube=False)
    home = reset_scene(planning_model)
    planning_arm = read_arm(planning_model)
    return Task(model, arm, gripper, cube, table, height, goal, planning_model, home, planning_arm)


def plan_pickup(
    task: Task, pickup: np.ndarray, *, yaw_offset: float = 0.0, start_angles: np.ndarray | None = None,
) -> Trajectory:
    _, orientation = tcp_pose(task.planning_home, task.planning_arm)
    grip_orientation = np.empty(4)
    mujoco.mju_mulQuat(grip_orientation, world_z_quaternion(yaw_offset), orientation)
    overhead = np.array([0.0, 0.0, 0.115])
    carried = np.array([0.0, 0.0, 0.09])
    targets = [
        PoseWaypoint(State.APPROACH.name, pickup + overhead, grip_orientation, 2.8),
        PoseWaypoint(State.DESCEND.name, pickup, grip_orientation, 2.4),
        PoseWaypoint(State.LIFT.name, pickup + carried, grip_orientation, 2.2),
        PoseWaypoint(State.TRANSPORT.name, task.goal + carried, orientation, 3.4),
        PoseWaypoint(State.PLACE.name, task.goal + [0, 0, 0.008], orientation, 2.2),
        PoseWaypoint(State.RETREAT.name, task.goal + overhead, orientation, 2.4),
    ]
    start = task.planning_home
    if start_angles is not None:
        # Continue from the last retreat on the cube-free planning model.
        start = mujoco.MjData(task.planning_model)
        mujoco.mj_copyData(start, task.planning_model, task.planning_home)
        start.qpos[task.planning_arm.qpos] = start_angles
        mujoco.mj_forward(task.planning_model, start)
    try:
        return plan_trajectory(task.planning_model, start, task.planning_arm, targets)
    except ValueError as error:
        if start_angles is None or not str(error).startswith("Joint reference exceeds velocity or acceleration limits:"):
            raise
        # The approach from a previous slot can be longer than the one from home.
        # Retime the references and run the same collision and joint-limit checks.
        slower = [replace(target, duration_s=1.5 * target.duration_s) for target in targets]
        return plan_trajectory(task.planning_model, start, task.planning_arm, slower)


class Contacts(NamedTuple):
    left_count: int
    right_count: int
    table_count: int
    left_normal_n: float
    right_normal_n: float
    table_normal_n: float
    finger_depth_m: float
    table_depth_m: float
    unexpected: str

    @property
    def gripping(self) -> bool:
        return min(self.left_normal_n, self.right_normal_n) >= MIN_NORMAL_N

    @property
    def supported(self) -> bool:
        return self.table_normal_n >= MIN_NORMAL_N


FINGER_CONTACT_STATES = {State.GRASP, State.LIFT, State.HOLD, State.TRANSPORT, State.PLACE}


def measure_contacts(task: Task, data: mujoco.MjData, state: State) -> Contacts:
    counts = [0, 0, 0]
    normal = [0.0, 0.0, 0.0]
    finger_depth = table_depth = 0.0
    unexpected = ""
    force = np.empty(6)
    model = task.model
    for i, contact in enumerate(data.contact):
        a, b = map(int, contact.geom)
        other = b if a == task.cube else a
        body = model.geom_bodyid[other]
        depth = max(0.0, -contact.dist)
        if task.cube in (a, b) and other == task.table:
            index = 2
            table_depth = max(table_depth, depth)
        elif task.table in (a, b) and (a in task.passive_cubes or b in task.passive_cubes):
            if depth > MAX_CONTACT_DEPTH_M:
                unexpected = f"passive cube / table ({1000 * depth:.3f} mm overlap)"
            continue
        elif task.cube in (a, b) and body in task.gripper.bodies and state in FINGER_CONTACT_STATES:
            index = task.gripper.bodies.index(body)
            finger_depth = max(finger_depth, depth)
        else:
            if contact.efc_address >= 0 or depth > 1e-6:
                labels = [model.geom(g).name or model.body(int(model.geom_bodyid[g])).name for g in (a, b)]
                unexpected = f"{labels[0]} / {labels[1]} ({1000 * depth:.3f} mm overlap)"
            continue
        if contact.efc_address >= 0:
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > 0:
                counts[index] += 1
                normal[index] += force[0]
    return Contacts(*counts, *normal, finger_depth, table_depth, unexpected)


@dataclass
class TrialResult:
    success: bool = False
    state: str = State.SETTLE.name
    reason: str = ""
    time_s: float = 0.0
    max_lift_m: float = 0.0
    hold_s: float = 0.0
    stable_s: float = 0.0
    min_hold_left_n: float | None = None
    min_hold_right_n: float | None = None
    max_finger_depth_m: float = 0.0
    max_table_depth_m: float = 0.0
    final_x_m: float = 0.0
    final_y_m: float = 0.0
    final_z_m: float = 0.0
    final_xy_error_m: float = 0.0
    initial_x_m: float = 0.0
    initial_y_m: float = 0.0
    initial_z_m: float = 0.0
    settled_x_m: float | None = None
    settled_y_m: float | None = None
    settled_z_m: float | None = None
    initial_yaw_deg: float = 0.0
    settled_yaw_deg: float | None = None
    grasp_yaw_offset_deg: float | None = None
    grasp_alignment_error_deg: float | None = None
    final_yaw_deg: float = 0.0
    final_square_yaw_error_deg: float = 0.0


class TrialFailure(RuntimeError):
    pass


def run_trial(
    task: Task, observe: Callable[[State, mujoco.MjData, Contacts], None] | None = None,
    *, cube_xy: np.ndarray | None = None, cube_yaw: float | None = None,
) -> TrialResult:
    """Start from the keyframe and execute each state through the actuators."""
    data = reset_scene(task.model, cube_xy, cube_yaw=cube_yaw)
    return execute_pick(task, data, observe, align_yaw=cube_yaw is not None)


def execute_pick(
    task: Task, data: mujoco.MjData,
    observe: Callable[[State, mujoco.MjData, Contacts], None] | None = None,
    *, align_yaw: bool, max_episode_steps: int | None = None,
    route_planner: Callable[[Task, mujoco.MjData, np.ndarray, float], Trajectory] | None = None,
) -> TrialResult:
    """Place one active cube using an existing live scene."""
    model, arm, gripper = task.model, task.arm, task.gripper
    dt = model.opt.timestep
    start_time = data.time
    remaining_steps = None if max_episode_steps is None else max_episode_steps - round(start_time / dt)
    home_angles = data.qpos[arm.qpos]
    stationary = np.zeros_like(home_angles)
    reference_step = 0
    opening = 1.0
    result = TrialResult()
    initial_xyz = data.geom_xpos[task.cube].copy()
    result.initial_x_m, result.initial_y_m, result.initial_z_m = map(float, initial_xyz)
    cube_rotation = data.geom_xmat[task.cube]
    jaw_rotation = data.xmat[arm.hand]
    result.initial_yaw_deg = degrees(planar_yaw(cube_rotation))
    cube_joint = model.body_jntadr[model.geom_bodyid[task.cube]]
    cube_dof = int(model.jnt_dofadr[cube_joint])
    steps = grasp_steps = hold_steps = stable_steps = 0
    settled_z = task.rest_z
    contacts = measure_contacts(task, data, State.SETTLE)
    placement_supported = False

    def at_goal() -> bool:
        return (
            np.linalg.norm(data.geom_xpos[task.cube, :2] - task.goal[:2]) <= GOAL_RADIUS_M
            and abs(data.geom_xpos[task.cube, 2] - task.rest_z) <= 0.0015
        )

    def on_goal() -> bool:
        return (
            contacts.supported and not (contacts.left_count or contacts.right_count) and at_goal()
            and (not align_yaw or abs(degrees(square_yaw_offset(planar_yaw(cube_rotation), 0.0))) <= GOAL_YAW_TOL_DEG)
        )

    def step(state: State, position: np.ndarray, velocity: np.ndarray, grip: float) -> None:
        nonlocal contacts, steps, grasp_steps, hold_steps, stable_steps, placement_supported
        if remaining_steps is not None and steps >= remaining_steps:
            raise TrialFailure(f"episode step budget ({max_episode_steps}) exhausted")
        data.ctrl[arm.actuators] = joint_controls(data, arm, position, velocity)
        gripper.command(data, grip)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        steps += 1
        if (
            abs(data.time - start_time - steps * dt) > dt / 4
            or any(warning.number for warning in data.warning)
            or not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all())
        ):
            raise TrialFailure("invalid dynamics, warning or simulation reset")
        angles = data.qpos[arm.qpos]
        if (np.any(angles < arm.joint_range[:, 0] - 1e-6)
                or np.any(angles > arm.joint_range[:, 1] + 1e-6)
                or np.max(np.abs(data.qvel[arm.dofs])) > MAX_TRACKING_SPEED):
            raise TrialFailure("arm joint angle or speed exceeded its limit")
        contacts = measure_contacts(task, data, state)
        result.max_finger_depth_m = max(result.max_finger_depth_m, contacts.finger_depth_m)
        result.max_table_depth_m = max(result.max_table_depth_m, contacts.table_depth_m)
        if observe is not None:
            observe(state, data, contacts)
        if contacts.unexpected:
            raise TrialFailure(f"unexpected contact {contacts.unexpected}")
        if max(contacts.finger_depth_m, contacts.table_depth_m) > MAX_CONTACT_DEPTH_M:
            raise TrialFailure("cube contact penetration exceeded 1 mm")

        height = float(data.geom_xpos[task.cube, 2]) - settled_z
        if state == State.GRASP:
            grasp_steps = grasp_steps + 1 if contacts.gripping else 0
        if state in (State.LIFT, State.HOLD, State.TRANSPORT):
            result.max_lift_m = max(result.max_lift_m, height)
        carrying = state in (State.LIFT, State.HOLD, State.TRANSPORT)
        lowering = state == State.PLACE and grip == 0.0
        if lowering and contacts.supported and at_goal():
            placement_supported = True
        if (carrying or lowering) and not contacts.gripping:
            # During load transfer a contact can disappear for one sample.
            # The cube must have already touched down inside the slot and stay there.
            if not (lowering and placement_supported and at_goal()):
                raise TrialFailure("lost load-bearing contact with one or both fingers")
        if state in (State.HOLD, State.TRANSPORT):
            if height < MIN_LIFT_M or contacts.table_count:
                raise TrialFailure("cube dropped below the required unsupported lift height")
        if state == State.HOLD:
            hold_steps += 1
            result.hold_s = hold_steps * dt
            left = float(contacts.left_normal_n)
            right = float(contacts.right_normal_n)
            result.min_hold_left_n = left if result.min_hold_left_n is None else min(result.min_hold_left_n, left)
            result.min_hold_right_n = right if result.min_hold_right_n is None else min(result.min_hold_right_n, right)
        if state == State.RETREAT:
            speed = data.qvel[cube_dof:cube_dof + 6]
            resting = on_goal() and np.linalg.norm(speed[:3]) < 0.005 and np.linalg.norm(speed[3:]) < 0.1
            stable_steps = stable_steps + 1 if resting else 0
            result.stable_s = stable_steps * dt

    def wait(state: State, angle: np.ndarray, duration: float, target_opening: float, ramp_s: float = 0) -> None:
        nonlocal opening
        for i in range(1, round(duration / dt) + 1):
            blend = min(1.0, i * dt / ramp_s) if ramp_s else 1.0
            step(state, angle, stationary, opening + blend * (target_opening - opening))
        opening = target_opening

    try:
        wait(State.SETTLE, home_angles, 0.8, 1.0)
        pickup = data.geom_xpos[task.cube].copy()
        result.settled_x_m, result.settled_y_m, result.settled_z_m = map(float, pickup)
        settled_yaw = planar_yaw(cube_rotation)
        result.settled_yaw_deg = degrees(settled_yaw)
        settled_z = float(pickup[2])
        expected = np.array([*initial_xyz[:2], task.rest_z])
        if not contacts.supported or np.linalg.norm(pickup - expected) > 0.001:
            raise TrialFailure("cube did not settle at pickup on the table")
        result.state = "PLAN"
        try:
            plan_options = {}
            yaw_offset = 0.0
            if align_yaw:
                closing_axis = planar_yaw(task.planning_home.xmat[task.planning_arm.hand], axis=1)
                yaw_offset = square_yaw_offset(settled_yaw, closing_axis)
                result.grasp_yaw_offset_deg = degrees(yaw_offset)
                plan_options["yaw_offset"] = yaw_offset
            if start_time > 0:
                plan_options["start_angles"] = data.qpos[arm.qpos]
            plan = (plan_pickup(task, pickup, **plan_options) if route_planner is None
                    else route_planner(task, data, pickup, yaw_offset))
        except (ValueError, RuntimeError) as error:
            raise TrialFailure(str(error)) from error
        waypoint_steps = dict(zip(plan.labels, plan.waypoint_steps))

        for state in tuple(State)[1:]:
            result.state = state.name
            if state == State.GRASP:
                result.grasp_alignment_error_deg = abs(degrees(square_yaw_offset(
                    planar_yaw(cube_rotation), planar_yaw(jaw_rotation, axis=1),
                )))
                wait(state, plan.angles[reference_step], 1.35, 0.0, ramp_s=0.75)
                if grasp_steps * dt < 0.15:
                    raise TrialFailure("bilateral load-bearing grasp did not last 0.15 s")
                continue
            if state == State.HOLD:
                wait(state, plan.angles[reference_step], 1.1, 0.0)
                if result.hold_s < MIN_HOLD_S:
                    raise TrialFailure("unsupported grasp hold did not reach 1 s")
                continue

            end = waypoint_steps[state.name]
            for i in range(reference_step, end):
                step(state, plan.angles[i], plan.velocities[i], opening)
            reference_step = end
            if state == State.LIFT:
                if data.geom_xpos[task.cube, 2] - settled_z < MIN_LIFT_M or contacts.table_count:
                    raise TrialFailure("cube did not lift at least 5 cm clear of the table")
            elif state == State.PLACE:
                wait(state, plan.angles[end], 1.2, 1.0, ramp_s=0.65)
                if not on_goal():
                    raise TrialFailure("released cube did not meet table support, target position or yaw tolerance")
            elif state == State.RETREAT:
                wait(state, plan.angles[end], (len(plan.angles) - 1 - end) * dt, 1.0)
                if result.stable_s < MIN_STABLE_S:
                    raise TrialFailure("released cube did not stay stable in the target for 1 s")
        result.success = True
    except TrialFailure as error:
        result.reason = str(error)

    final_xyz = data.geom_xpos[task.cube]
    result.final_x_m, result.final_y_m, result.final_z_m = map(float, final_xyz)
    result.final_yaw_deg = degrees(planar_yaw(cube_rotation))
    result.final_square_yaw_error_deg = abs(degrees(square_yaw_offset(planar_yaw(cube_rotation), 0.0)))
    result.final_xy_error_m = float(np.linalg.norm(final_xyz[:2] - task.goal[:2]))
    result.time_s = float(data.time - start_time)
    return result


def run_batch(
    task: Task, count: int, output_dir: Path, *,
    video: bool = False, positions: np.ndarray | None = None, yaws: np.ndarray | None = None,
    seed: int | None = None,
) -> list[TrialResult]:
    if positions is not None and positions.shape != (count, 2):
        raise ValueError("One cube XY position is required per trial")
    if yaws is not None and yaws.shape != (count,):
        raise ValueError("One cube yaw is required per trial")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "panda_pick_place.mp4"
    cube_body = int(task.model.geom_bodyid[task.cube])
    results = []
    with (output_dir / "tracking.csv").open("w", newline="", encoding="utf-8") as tracking_file, (
        output_dir / "trials.csv"
    ).open("w", newline="", encoding="utf-8") as results_file:
        tracking = csv.writer(tracking_file)
        tracking.writerow((
            "trial", "state", "time_s", "gripper_ctrl", "gripper_tendon_force_n", "left_slide_m", "right_slide_m",
            "cube_x_m", "cube_y_m", "cube_z_m", "cube_qw", "cube_qx", "cube_qy", "cube_qz",
            "tcp_x_m", "tcp_y_m", "tcp_z_m", "cube_yaw_deg", "jaw_yaw_deg", *Contacts._fields,
        ))
        summary = csv.DictWriter(
            results_file, fieldnames=["trial", "seed", *(field.name for field in fields(TrialResult))],
        )
        summary.writeheader()
        for trial in range(1, count + 1):
            cube_xy = None if positions is None else positions[trial - 1]
            cube_yaw = None if yaws is None else float(yaws[trial - 1])
            if trial == 1 and video:
                from panda_pick_place_video import record_video

                caption = ""
                if cube_xy is not None:
                    caption = f"Trial {trial:02d}   |   Start XY  ({cube_xy[0]:.3f}, {cube_xy[1]:.3f}) m"
                    if cube_yaw is not None:
                        caption = f"Start XY ({cube_xy[0]:.3f}, {cube_xy[1]:.3f}) m  |  Yaw {degrees(cube_yaw):+.1f} deg"
                recording = record_video(task.model, video_path, caption=caption)
            else:
                recording = nullcontext(None)

            with recording as capture:
                def observe(state: State, data: mujoco.MjData, contacts: Contacts) -> None:
                    grip = data.ctrl[task.gripper.actuator]
                    xyz = tcp_position(data, task.arm)
                    values = (
                        grip, data.actuator_force[task.gripper.actuator],
                        *(data.qpos[i] for i in task.gripper.fingers),
                        *data.geom_xpos[task.cube], *data.xquat[cube_body], *xyz,
                        degrees(planar_yaw(data.geom_xmat[task.cube])),
                        degrees(planar_yaw(data.xmat[task.arm.hand], axis=1)),
                    )
                    tracking.writerow((trial, state.name, f"{data.time:.3f}",
                                       *(f"{value:.9g}" for value in values), *contacts))
                    if capture is not None:
                        capture(state.name, data, contacts.left_normal_n, contacts.right_normal_n)

                result = run_trial(task, observe, cube_xy=cube_xy, cube_yaw=cube_yaw)
            results.append(result)
            summary.writerow({"trial": trial, "seed": seed, **asdict(result)})
            results_file.flush()
            tracking_file.flush()
            outcome = "PASS" if result.success else f"FAIL at {result.state}: {result.reason}"
            yaw_label = f", {result.initial_yaw_deg:+.1f} deg" if cube_yaw is not None else ""
            print(f"Trial {trial:02d} {outcome} | start ({result.initial_x_m:.4f}, {result.initial_y_m:.4f}) m{yaw_label} | "
                  f"lift {1000 * result.max_lift_m:.2f} mm | "
                  f"hold {result.hold_s:.2f} s | goal error {1000 * result.final_xy_error_m:.3f} mm", flush=True)
    return results


def plot_positions(task: Task, results: list[TrialResult], path: Path, *, seed: int | None) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    figure = Figure(figsize=(7, 6.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    table = task.model.geom(task.table)
    table_xy, table_half_size = table.pos[:2], table.size[:2]
    ax.add_patch(Rectangle(table_xy - table_half_size, *(2 * table_half_size),
                           facecolor="#f1f3f6", edgecolor="#b5bdc9", label="Tabletop"))
    region_size = np.subtract(SPAWN_MAX_XY, SPAWN_MIN_XY)
    ax.add_patch(Rectangle(SPAWN_MIN_XY, *region_size, facecolor="#e3f2ed", edgecolor="#438b77",
                           linestyle="--", label="Spawn region"))
    xy = np.array([(result.initial_x_m, result.initial_y_m) for result in results])
    passed = np.array([result.success for result in results])
    for mask, color, marker, label in (
        (passed, "#007a65", "o", "Success"), (~passed, "#c13f3c", "X", "Failure"),
    ):
        if mask.any():
            ax.scatter(*xy[mask].T, s=42, color=color, marker=marker, zorder=3, label=label)
    if seed is None:
        for i, (x, y) in enumerate(xy, 1):
            ax.annotate(str(i), (x, y), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.scatter(*task.goal[:2], marker="*", s=165, color="#2459a6", zorder=4, label="Fixed place goal")
    title = "Five validation positions" if seed is None else f"Random pickup positions  |  seed {seed}"
    ax.set(title=f"{title}\n{sum(passed)}/{len(results)} successful",
           xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.autoscale_view()
    figure.savefig(path, dpi=160)


def plot_yaws(results: list[TrialResult], path: Path, *, seed: int | None) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(10, 4.3), layout="constrained")
    FigureCanvasAgg(figure)
    histogram, adjustments = figure.subplots(1, 2)
    initial = np.array([result.initial_yaw_deg for result in results])
    offsets = np.array([result.grasp_yaw_offset_deg for result in results], dtype=float)
    passed = np.array([result.success for result in results])
    histogram.hist((initial[passed], initial[~passed]), bins=np.arange(-180, 181, 30),
                   stacked=True, color=["#007a65", "#c13f3c"], label=["Success", "Failure"], rwidth=0.88)
    for mask, color, marker in ((passed, "#007a65", "o"), (~passed, "#c13f3c", "X")):
        adjustments.scatter(initial[mask], offsets[mask], s=40, c=color, marker=marker, zorder=3)
    histogram.set(title="Initial cube yaw distribution", ylabel="Trials")
    histogram.yaxis.set_major_locator(MaxNLocator(integer=True))
    histogram.legend()
    adjustments.set(title="Yaw correction used for grasp", ylabel="Gripper rotation from home (deg)", ylim=(-49, 49))
    adjustments.axhline(0, color="#56677a", linewidth=0.9)
    for ax in (histogram, adjustments):
        ax.set(xlabel="Initial cube yaw (deg)", xlim=(-180, 180), xticks=np.arange(-180, 181, 60))
        ax.grid(alpha=0.25)
    if seed is not None:
        figure.suptitle(f"Random pickup yaw  |  seed {seed}")
    figure.savefig(path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    output_names = {
        "fixed": "pick_place", "validation": "random_validation", "random": "random_pick_place",
        "yaw-validation": "yaw_validation", "random-yaw": "random_yaw_pick_place",
    }
    parser.add_argument("--mode", choices=output_names, default="fixed",
                        help="validate or randomize XY alone, or XY plus cube yaw")
    parser.add_argument("--trials", type=int, help="number of independent resets (fixed 10, validation 5, random 30)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="nonnegative PCG64 seed for random mode")
    parser.add_argument("--video", action="store_true", help="record the first trial as an MP4")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    validation = args.mode in ("validation", "yaw-validation")
    random = args.mode in ("random", "random-yaw")
    default_trials = 5 if validation else 30 if random else 10
    count = default_trials if args.trials is None else args.trials
    if count < 1:
        parser.error("--trials must be at least 1")
    if validation and count != 5:
        parser.error("validation modes always run five trials")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    seed = args.seed if random else None
    positions = None
    yaws = None
    if args.mode == "validation":
        positions = validation_pickups()
    elif args.mode == "random":
        positions = sample_pickups(count, args.seed)
    elif args.mode in ("yaw-validation", "random-yaw"):
        poses = validation_poses() if validation else sample_pickup_poses(count, args.seed)
        positions, yaws = poses[:, :2], poses[:, 2]
    output_dir = args.output_dir or ROOT / "outputs" / output_names[args.mode]

    task = prepare_task()
    print(f"MuJoCo {mujoco.__version__} | {task.model.opt.timestep:.3f} s timestep | "
          f"mode {args.mode} | seed {seed} | place {task.goal.round(4)} m", flush=True)
    results = run_batch(task, count, output_dir, video=args.video, positions=positions, yaws=yaws, seed=seed)
    if positions is not None:
        plot_positions(task, results, output_dir / "positions.png", seed=seed)
    if yaws is not None:
        plot_yaws(results, output_dir / "yaw_distribution.png", seed=seed)
    successes = sum(result.success for result in results)
    rate = successes / count
    print(f"Success rate: {successes}/{count} ({rate:.1%})")
    print(f"Results: {output_dir.resolve()}")
    threshold = 1.0 if validation else 0.8
    if rate < threshold:
        parser.exit(1, f"Success rate below {threshold:.0%}\n")


if __name__ == "__main__":
    main()
