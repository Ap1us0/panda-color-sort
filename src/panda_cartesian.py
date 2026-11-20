"""Plan and track a collision-checked six-dimensional TCP path for the MuJoCo Panda."""

import argparse
import csv
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import NamedTuple

import mujoco
import numpy as np
from numpy.typing import NDArray

from panda_position import ROOT, load_panda, require_position_servo


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
WAYPOINT_LABELS = ("home", "horizontal", "lower", "raise", "second target")
# The finger pad center is ~0.0584 + 0.0445 m along the hand's local Z axis.
TCP_OFFSET_M = np.array([0.0, 0.0, 0.1034])
WAYPOINT_OFFSETS_M = np.array([
    [0.0, 0.0, 0.0],
    [0.045, 0.12, 0.0],
    [0.045, 0.12, -0.13],
    [0.045, 0.12, 0.0],
    [-0.09, 0.20, 0.055],
])
SEGMENT_DURATIONS_S = (2.2, 2.0, 2.0, 2.5)
FINAL_HOLD_S = 1.3
JOINT_MARGIN_RAD = 0.05
MAX_REFERENCE_SPEED = 0.8  # rad/s
MAX_REFERENCE_ACCEL = 1.8  # rad/s²
MAX_TRACKING_SPEED = 1.0  # rad/s
MAX_TRACKING_ACCEL = 2.5  # rad/s²
ORIENTATION_WEIGHT = 0.25  # meters per radian in the weighted pose error
IK_DAMPING = 0.01
SINGULAR_VALUE_THRESHOLD = 0.06
NEAR_SINGULAR_DAMPING = 0.12
IK_TOLERANCE = 1e-8
FINAL_ERROR_LIMIT_M = 0.02
ORIENTATION_ERROR_LIMIT_DEG = 5.0
DEMO_ROTATION_DEG = 25.0

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Arm:
    hand: int
    qpos: NDArray[np.intp]
    dofs: NDArray[np.intp]
    actuators: NDArray[np.intp]
    joint_range: FloatArray
    ctrl_range: FloatArray
    torque_range: FloatArray
    kp: FloatArray
    kd: FloatArray


def read_arm(model: mujoco.MjModel) -> Arm:
    joints = np.array([model.joint(name).id for name in JOINT_NAMES])
    actuators = np.array([model.actuator(f"actuator{i}").id for i in range(1, 8)])
    for joint, actuator in zip(joints, actuators):
        require_position_servo(model, model.joint(int(joint)).name, model.actuator(int(actuator)).name)
    if not (
        model.jnt_limited[joints].all()
        and model.actuator_ctrllimited[actuators].all()
        and model.actuator_forcelimited[actuators].all()
    ):
        raise ValueError("All seven arm joints must have position, control, and output limits")
    return Arm(
        model.body("hand").id,
        model.jnt_qposadr[joints],
        model.jnt_dofadr[joints],
        actuators,
        model.jnt_range[joints],
        model.actuator_ctrlrange[actuators],
        model.actuator_forcerange[actuators],
        model.actuator_gainprm[actuators, 0],
        -model.actuator_biasprm[actuators, 2],
    )


def tcp_position(data: mujoco.MjData, arm: Arm) -> FloatArray:
    return data.xpos[arm.hand] + data.xmat[arm.hand].reshape(3, 3) @ TCP_OFFSET_M


def tcp_pose(data: mujoco.MjData, arm: Arm) -> tuple[FloatArray, FloatArray]:
    """Return world XYZ in meters and the tool-to-world quaternion in wxyz order."""
    return tcp_position(data, arm), data.xquat[arm.hand].copy()


def unit_quaternion(orientation: FloatArray) -> FloatArray:
    quaternion = np.asarray(orientation, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.any(quaternion):
        raise ValueError("Orientation must be a finite nonzero wxyz quaternion")
    scaled = quaternion / np.max(np.abs(quaternion))
    return scaled / np.linalg.norm(scaled)


def rotation_error(target: FloatArray, actual: FloatArray) -> FloatArray:
    """Shortest rotation vector from actual to target, in world axes (radians)."""
    inverse = actual * (1, -1, -1, -1)
    relative = np.empty(4)
    mujoco.mju_mulQuat(relative, target, inverse)
    if relative[0] < 0:
        relative *= -1
    error = np.empty(3)
    mujoco.mju_quat2Vel(error, relative, 1.0)
    return error


def damped_step(jacobian: FloatArray, error: FloatArray) -> FloatArray:
    u, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    proximity = max(0.0, 1.0 - singular_values[-1] / SINGULAR_VALUE_THRESHOLD)
    damping = IK_DAMPING + NEAR_SINGULAR_DAMPING * proximity**2
    gains = singular_values / (singular_values**2 + damping**2)
    return vt.T @ (gains * (u.T @ error))


def check_contacts(model: mujoco.MjModel, data: mujoco.MjData, stage: str) -> None:
    for contact in data.contact:
        if contact.dist < -1e-6:
            bodies = [model.body(int(model.geom_bodyid[geom])).name for geom in contact.geom]
            raise RuntimeError(f"{stage}: collision {bodies[0]} / {bodies[1]} (dist={contact.dist:.6f} m)")


class InverseKinematics:
    def __init__(self, model: mujoco.MjModel, home: mujoco.MjData, arm: Arm):
        self.model = model
        self.arm = arm
        # Candidate poses live in separate data. Only mj_step moves the simulated arm.
        self.scratch = mujoco.MjData(model)
        mujoco.mj_copyData(self.scratch, model, home)
        self.home_orientation = home.xquat[arm.hand].copy()
        self.lower = np.maximum(arm.joint_range[:, 0], arm.ctrl_range[:, 0]) + JOINT_MARGIN_RAD
        self.upper = np.minimum(arm.joint_range[:, 1], arm.ctrl_range[:, 1]) - JOINT_MARGIN_RAD
        self.jacp = np.empty((3, model.nv))
        self.jacr = np.empty((3, model.nv))

    def pose_error(
        self, angles: FloatArray, target: FloatArray, orientation: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        self.scratch.qpos[self.arm.qpos] = angles
        mujoco.mj_fwdPosition(self.model, self.scratch)
        point = tcp_position(self.scratch, self.arm)
        angular_error = rotation_error(orientation, self.scratch.xquat[self.arm.hand])
        error = np.concatenate((target - point, ORIENTATION_WEIGHT * angular_error))
        return point, error

    def solve(
        self, target: FloatArray, seed: FloatArray, *,
        orientation: FloatArray | None = None, stage: str = "IK",
    ) -> FloatArray:
        target = np.asarray(target, dtype=float)
        seed = np.asarray(seed, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("IK target must be a finite world XYZ vector")
        if seed.shape != self.lower.shape or not np.isfinite(seed).all():
            raise ValueError("IK seed must contain seven finite joint angles")
        orientation = self.home_orientation if orientation is None else unit_quaternion(orientation)
        angles = np.clip(seed, self.lower, self.upper)
        point, error = self.pose_error(angles, target, orientation)
        for _ in range(120):
            norm = np.linalg.norm(error)
            if norm < IK_TOLERANCE:
                check_contacts(self.model, self.scratch, stage)
                return angles
            mujoco.mj_jac(self.model, self.scratch, self.jacp, self.jacr, point, self.arm.hand)
            jac = np.vstack((self.jacp[:, self.arm.dofs], ORIENTATION_WEIGHT * self.jacr[:, self.arm.dofs]))
            delta = damped_step(jac, error)
            delta *= min(1.0, 0.12 / max(0.12, np.max(np.abs(delta))))
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = np.clip(angles + fraction * delta, self.lower, self.upper)
                trial_point, trial_error = self.pose_error(trial, target, orientation)
                if np.linalg.norm(trial_error) < norm:
                    angles, point, error = trial, trial_point, trial_error
                    break
            else:
                break
        position_residual = np.linalg.norm(error[:3])
        angle_residual = np.rad2deg(np.linalg.norm(error[3:]) / ORIENTATION_WEIGHT)
        raise ValueError(f"{stage}: unreachable or IK did not converge within joint limits; "
                         f"XYZ residual {position_residual:.6f} m, orientation residual {angle_residual:.4f} deg")


@dataclass(frozen=True)
class PoseWaypoint:
    label: str
    xyz: FloatArray
    quaternion: FloatArray
    duration_s: float  # travel time from the previous waypoint


def demo_waypoints(data: mujoco.MjData, arm: Arm, *, hold_orientation: bool = False) -> list[PoseWaypoint]:
    position, orientation = tcp_pose(data, arm)
    final_orientation = orientation.copy()
    if not hold_orientation:
        axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        relative = np.empty(4)
        mujoco.mju_axisAngle2Quat(relative, axis, np.deg2rad(DEMO_ROTATION_DEG))
        mujoco.mju_mulQuat(final_orientation, relative, orientation)
    return [
        PoseWaypoint(label, position + offset, quaternion, duration)
        for label, offset, quaternion, duration in zip(
            WAYPOINT_LABELS[1:], WAYPOINT_OFFSETS_M[1:],
            (orientation, orientation, orientation, final_orientation), SEGMENT_DURATIONS_S,
        )
    ]


def interpolate_orientation(start: FloatArray, end: FloatArray, blend: FloatArray) -> FloatArray:
    # The SO(3) exponential traces the shortest quaternion arc (SLERP).
    rotation = rotation_error(end, start)
    angle = np.linalg.norm(rotation)
    if angle == 0:
        return np.repeat(start[None, :], len(blend), axis=0)
    axis = rotation / angle
    relative = np.empty(4)
    quaternions = np.empty((len(blend), 4))
    for quaternion, fraction in zip(quaternions, blend):
        mujoco.mju_axisAngle2Quat(relative, axis, angle * fraction)
        mujoco.mju_mulQuat(quaternion, relative, start)
    return quaternions


@dataclass(frozen=True)
class Trajectory:
    waypoints: FloatArray
    labels: tuple[str, ...]
    waypoint_steps: NDArray[np.intp]
    xyz: FloatArray
    quaternions: FloatArray
    angles: FloatArray
    velocities: FloatArray
    peak_speed: float
    peak_accel: float


def plan_trajectory(
    model: mujoco.MjModel, home: mujoco.MjData, arm: Arm,
    targets: Sequence[PoseWaypoint] | None = None,
) -> Trajectory:
    dt = model.opt.timestep
    targets = demo_waypoints(home, arm) if targets is None else targets
    if len(targets) == 0:
        raise ValueError("At least one pose target is required")
    home_xyz, home_quaternion = tcp_pose(home, arm)
    waypoints = np.array([home_xyz, *(target.xyz for target in targets)], dtype=float)
    if waypoints.shape != (len(targets) + 1, 3) or not np.isfinite(waypoints).all():
        raise ValueError("Each waypoint requires a finite world XYZ vector")
    durations = np.array([target.duration_s for target in targets])
    if not np.isfinite(durations).all() or np.any(durations < dt):
        raise ValueError("Each segment duration must be finite and at least one physics timestep")
    labels = ("home", *(target.label for target in targets))
    segment_steps = np.round(durations / dt).astype(int)
    waypoint_steps = np.r_[0, np.cumsum(segment_steps)]
    ik = InverseKinematics(model, home, arm)
    xyz_segments = [waypoints[:1]]
    quaternion_segments = [home_quaternion[None, :]]
    for start, end, target, steps in zip(waypoints[:-1], waypoints[1:], targets, segment_steps):
        # The same quintic progress gives zero linear and angular velocity
        # and acceleration at the waypoint junctions.
        u = np.arange(1, steps + 1) / steps
        blend = u**3 * (10 + u * (-15 + 6 * u))
        xyz_segments.append(start + blend[:, None] * (end - start))
        quaternion_segments.append(interpolate_orientation(
            quaternion_segments[-1][-1], unit_quaternion(target.quaternion), blend,
        ))
    xyz = np.concatenate(xyz_segments)
    quaternions = np.concatenate(quaternion_segments)
    angles = np.empty((len(xyz), len(JOINT_NAMES)))
    seed = home.qpos[arm.qpos]
    waypoint_labels = dict(zip(waypoint_steps, labels))
    for step, (target, orientation) in enumerate(zip(xyz, quaternions)):
        stage = f"path at {step * dt:.3f} s"
        if step in waypoint_labels:
            stage += f" ({waypoint_labels[step]})"
        seed = ik.solve(target, seed, orientation=orientation, stage=stage)
        angles[step] = seed

    speeds = np.diff(angles, axis=0) / dt
    peak_speed = float(np.max(np.abs(speeds)))
    peak_accel = float(np.max(np.abs(np.diff(speeds, axis=0, prepend=0.0, append=0.0))) / dt)
    if peak_speed > MAX_REFERENCE_SPEED or peak_accel > MAX_REFERENCE_ACCEL:
        raise ValueError(f"Joint reference exceeds velocity or acceleration limits: {peak_speed:.3f} rad/s, "
                         f"{peak_accel:.3f} rad/s^2")

    hold = round(FINAL_HOLD_S / dt)
    xyz = np.concatenate((xyz, np.repeat(xyz[-1:], hold, axis=0)))
    quaternions = np.concatenate((quaternions, np.repeat(quaternions[-1:], hold, axis=0)))
    angles = np.concatenate((angles, np.repeat(angles[-1:], hold, axis=0)))
    velocities = np.gradient(angles, dt, axis=0)
    velocities[0] = 0.0
    return Trajectory(waypoints, labels, waypoint_steps, xyz, quaternions, angles, velocities, peak_speed, peak_accel)


class TrackingRun(NamedTuple):
    times: FloatArray
    xyz: FloatArray
    quaternions: FloatArray
    angles: FloatArray
    velocities: FloatArray
    torques: FloatArray


def joint_controls(data: mujoco.MjData, arm: Arm, angle: FloatArray, velocity: FloatArray) -> FloatArray:
    position = data.qpos[arm.qpos]
    speed = data.qvel[arm.dofs]
    torque = arm.kp * (angle - position) + arm.kd * (velocity - speed) + data.qfrc_bias[arm.dofs]
    torque = np.clip(torque, arm.torque_range[:, 0], arm.torque_range[:, 1])
    # Invert the verified general actuator law: tau = kp * (ctrl - q) - kd * dq.
    control = position + (torque + arm.kd * speed) / arm.kp
    return np.clip(control, arm.ctrl_range[:, 0], arm.ctrl_range[:, 1])


def add_connector(scene: mujoco.MjvScene, start: FloatArray, end: FloatArray, *, measured: bool) -> None:
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(end - start) < 1e-5:
        return
    kind = mujoco.mjtGeom.mjGEOM_CAPSULE if measured else mujoco.mjtGeom.mjGEOM_LINE
    width = 0.0024 if measured else 3.0
    color = (1.0, 0.40, 0.05, 1.0) if measured else (0.06, 0.92, 1.0, 1.0)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
    mujoco.mjv_connector(geom, kind, width, start, end)
    scene.ngeom += 1


def draw_waypoints(scene: mujoco.MjvScene, points: FloatArray) -> None:
    for start, end in zip(points[:-1], points[1:]):
        add_connector(scene, start, end, measured=False)
    for point in points:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, (0.006, 0, 0), point,
            np.eye(3).ravel(), (0.06, 0.92, 1.0, 1.0),
        )
        scene.ngeom += 1


def draw_tool_axes(scene: mujoco.MjvScene, point: FloatArray, rotation: FloatArray) -> None:
    kind = mujoco.mjtGeom.mjGEOM_ARROW
    colors = ((1.0, 0.22, 0.18, 1.0), (0.22, 0.9, 0.32, 1.0), (0.2, 0.45, 1.0, 1.0))
    for i, (axis, color) in enumerate(zip(rotation.reshape(3, 3).T, colors)):
        geom = scene.geoms[i]  # The first three slots are reserved when opening the viewer.
        mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, kind, 0.003, point, point + 0.08 * axis)


def track_trajectory(
    model: mujoco.MjModel, data: mujoco.MjData, arm: Arm, plan: Trajectory, *, viewer: bool = False
) -> TrackingRun:
    dt = model.opt.timestep
    count = len(plan.xyz)
    run = TrackingRun(np.empty(count), np.empty((count, 3)), np.empty((count, 4)), np.empty_like(plan.angles),
                      np.empty_like(plan.angles), np.zeros_like(plan.angles))
    view_context = nullcontext()
    if viewer:
        from mujoco.viewer import launch_passive

        view_data = mujoco.MjData(model)
        mujoco.mj_copyData(view_data, model, data)
        view_context = launch_passive(model, view_data, show_left_ui=False, show_right_ui=False)

    with view_context as window:
        start = 0.0
        frame_steps = max(1, round(1 / (40 * dt)))
        trail_point = tcp_position(data, arm)
        if window is not None:
            with window.lock():
                window.cam.lookat[:] = (0.36, 0.06, 0.44)
                window.cam.distance = 1.85
                window.cam.azimuth = 120
                window.cam.elevation = -27
                window.user_scn.ngeom = 3
                draw_tool_axes(window.user_scn, trail_point, data.xmat[arm.hand])
                draw_waypoints(window.user_scn, plan.waypoints)
            window.sync()
            start = perf_counter() + 1.5

        for i in range(count):
            if i:
                data.ctrl[arm.actuators] = joint_controls(data, arm, plan.angles[i - 1], plan.velocities[i - 1])
                mujoco.mj_step(model, data)
                run.torques[i] = data.actuator_force[arm.actuators]
                # Refresh world kinematics and contacts for the integrated state.
                mujoco.mj_forward(model, data)

            stage = f"simulation at {i * dt:.3f} s"
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                    and np.isfinite(run.torques[i]).all()
                    and np.isclose(data.time, i * dt, atol=dt / 4, rtol=0)
                    and not any(warning.number for warning in data.warning)):
                raise RuntimeError(f"{stage}: invalid or reset dynamics state")
            check_contacts(model, data, stage)
            run.times[i] = data.time
            run.xyz[i], run.quaternions[i] = tcp_pose(data, arm)
            run.angles[i] = data.qpos[arm.qpos]
            run.velocities[i] = data.qvel[arm.dofs]
            if (np.any(run.angles[i] < arm.joint_range[:, 0] - 1e-6)
                    or np.any(run.angles[i] > arm.joint_range[:, 1] + 1e-6)
                    or np.max(np.abs(run.velocities[i])) > MAX_TRACKING_SPEED
                    or (i and np.max(np.abs(run.velocities[i] - run.velocities[i - 1])) / dt > MAX_TRACKING_ACCEL)):
                raise RuntimeError(f"{stage}: measured joint position, speed, or acceleration limit exceeded")

            if window is not None and window.is_running() and (i % frame_steps == 0 or i == count - 1):
                sleep(max(0.0, start + data.time - perf_counter()))
                with window.lock():
                    mujoco.mj_copyData(view_data, model, data)
                    add_connector(window.user_scn, trail_point, run.xyz[i], measured=True)
                    draw_tool_axes(window.user_scn, run.xyz[i], data.xmat[arm.hand])
                window.sync(state_only=True)
                trail_point = run.xyz[i]
    return run


def tracking_errors(plan: Trajectory, run: TrackingRun) -> tuple[FloatArray, FloatArray]:
    rotation_vectors = np.array([
        rotation_error(target, actual) for target, actual in zip(plan.quaternions, run.quaternions)
    ])
    return plan.xyz - run.xyz, np.rad2deg(rotation_vectors)


def save_results(plan: Trajectory, run: TrackingRun, output_dir: Path) -> tuple[Path, Path, Path]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "panda_cartesian_tracking.csv"
    error_path = figures / "panda_cartesian_pose_error.png"
    angles_path = figures / "panda_cartesian_joint_angles.png"

    xyz_error, orientation_error = tracking_errors(plan, run)
    error_norm = np.linalg.norm(xyz_error, axis=1)
    angle_error_deg = np.linalg.norm(orientation_error, axis=1)
    columns = ["time_s"]
    for kind in ("target", "actual", "error"):
        columns.extend(f"{kind}_{axis}_m" for axis in "xyz")
    columns.append("error_norm_m")
    for kind, unit in (("target", "rad"), ("actual", "rad"), ("velocity", "rad_s"), ("applied_torque", "nm")):
        columns.extend(f"{kind}_{joint}_{unit}" for joint in JOINT_NAMES)
    for kind in ("target", "actual"):
        columns.extend(f"{kind}_quat_{axis}" for axis in "wxyz")
    columns.extend(f"orientation_error_{axis}_deg" for axis in "xyz")
    columns.append("orientation_error_deg")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(np.column_stack((run.times, plan.xyz, run.xyz, xyz_error, error_norm,
                                          plan.angles, run.angles, run.velocities, run.torques,
                                          plan.quaternions, run.quaternions, orientation_error, angle_error_deg)))

    figure = Figure(figsize=(10, 8), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle("Panda six-dimensional TCP tracking error")
    axes = figure.subplots(3, 2, sharex=True).ravel()
    for ax, values, name, unit, color in zip(
        axes[:5], (*xyz_error.T * 1000, error_norm * 1000, angle_error_deg),
        ("World X", "World Y", "World Z", "Position norm", "Orientation angle"),
        ("mm", "mm", "mm", "mm", "deg"),
        ("#1568a0", "#16856d", "#973f93", "#b14e17", "#b14e17"),
    ):
        ax.plot(run.times, values, color=color)
        ax.set(title=name, ylabel=f"Error ({unit})")
        ax.grid(alpha=0.25)
    for axis, values, color in zip("XYZ", orientation_error.T, ("#1568a0", "#16856d", "#973f93")):
        axes[-1].plot(run.times, values, label=axis, color=color)
    axes[-1].set(title="World rotation-vector components", ylabel="Error (deg)")
    axes[-1].legend(loc="upper left", frameon=False, ncols=3)
    axes[-1].grid(alpha=0.25)
    for ax in axes:
        for time_s in run.times[plan.waypoint_steps[1:]]:
            ax.axvline(time_s, color="#687280", linestyle=":", linewidth=0.8, alpha=0.65)
    figure.supxlabel("Simulation time (s)")
    figure.savefig(error_path, dpi=150)

    figure = Figure(figsize=(10, 8.5), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle("Panda arm joint angles")
    axes = figure.subplots(4, 2, sharex=True).ravel()
    for i, joint in enumerate(JOINT_NAMES):
        ax = axes[i]
        ax.plot(run.times, run.angles[:, i], color="#bd531e", label="Measured", linewidth=2.2)
        ax.plot(run.times, plan.angles[:, i], "--", color="#174b7a", label="IK target", linewidth=1.0)
        ax.set(title=joint, ylabel="Angle (rad)")
        ax.grid(alpha=0.25)
    axes[-1].legend(*axes[0].get_legend_handles_labels(), loc="center", frameon=False)
    axes[-1].set_axis_off()
    figure.supxlabel("Simulation time (s)")
    figure.savefig(angles_path, dpi=150)
    return csv_path, error_path, angles_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="show the target path and live measured trail")
    parser.add_argument("--hold-orientation", action="store_true", help="keep the home orientation on every segment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="directory for CSV and figures")
    args = parser.parse_args()
    try:
        model, data = load_panda()
        arm = read_arm(model)
        plan = plan_trajectory(model, data, arm, demo_waypoints(data, arm, hold_orientation=args.hold_orientation))
        print(f"MuJoCo {mujoco.__version__} | {len(arm.actuators)} arm actuators / {model.nu} total", flush=True)
        print(f"Preflight passed | peak reference speed {plan.peak_speed:.3f} rad/s | "
              f"peak acceleration {plan.peak_accel:.3f} rad/s^2", flush=True)
        run = track_trajectory(model, data, arm, plan, viewer=args.viewer)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    xyz_error, orientation_error = tracking_errors(plan, run)
    position_errors = np.linalg.norm(xyz_error, axis=1)
    angle_errors = np.linalg.norm(orientation_error, axis=1)
    print("Targets and measured error at waypoint arrival:")
    for label, i in zip(plan.labels, plan.waypoint_steps):
        print(f"  {run.times[i]:5.2f} s {label:14} XYZ {plan.xyz[i].round(4)} m | "
              f"wxyz {plan.quaternions[i].round(5)} | {1000 * position_errors[i]:.4f} mm, {angle_errors[i]:.4f} deg")
    for path in save_results(plan, run, args.output_dir):
        print(path.resolve())
    print(f"{len(run.times) - 1} physics steps | {run.times[-1]:.3f} s")
    print(f"Position error | final {1000 * position_errors[-1]:.6f} mm | peak {1000 * position_errors.max():.4f} mm "
          f"| RMS {1000 * np.sqrt(np.mean(position_errors**2)):.4f} mm")
    print(f"Orientation error | final {angle_errors[-1]:.6f} deg | peak {angle_errors.max():.4f} deg "
          f"| RMS {np.sqrt(np.mean(angle_errors**2)):.4f} deg")
    print(f"Measured joint excursions (rad) {np.ptp(run.angles, axis=0).round(4)}")
    print(f"Measured peak joint speed {np.max(np.abs(run.velocities)):.3f} rad/s | "
          f"peak acceleration {np.max(np.abs(np.diff(run.velocities, axis=0))) / model.opt.timestep:.3f} rad/s^2")
    if position_errors.max() >= FINAL_ERROR_LIMIT_M or angle_errors.max() >= ORIENTATION_ERROR_LIMIT_DEG:
        parser.exit(1, "Error: tracking must stay below 20 mm and 5 deg over the entire trajectory\n")


if __name__ == "__main__":
    main()
