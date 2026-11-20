"""Run position-control experiments on joint1 of the MuJoCo Menagerie Panda."""

import argparse
import csv
import math
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter, sleep
from typing import NamedTuple

import mujoco


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "models" / "franka_emika_panda" / "scene.xml"
JOINT = "joint1"
ACTUATOR = "actuator1"
STEP_TIME_S = 0.5
DURATION_S = 3.0
DEFAULT_TARGET_RAD = 0.5
TRACKING_AMPLITUDE_RAD = 0.8
TRACKING_PERIOD_S = 4.0
TRACKING_DURATION_S = 2 * TRACKING_PERIOD_S
VIEWER_SPEED = 0.2
TRACKING_VIEWER_SPEED = 0.5
VIEWER_LEAD_IN_S = 2.0


class Sample(NamedTuple):
    time_s: float
    target_rad: float
    angle_rad: float

    @property
    def error_rad(self) -> float:
        return self.target_rad - self.angle_rad


def load_panda_spec(scene: Path = SCENE) -> mujoco.MjSpec:
    # Python handles Unicode Windows paths. MuJoCo resolves the relative model
    # resources from memory, independently of the current working directory.
    directory = SCENE.parent
    assets = {
        f"assets/{file.name}": file.read_bytes()
        for file in (directory / "assets").iterdir() if file.is_file()
    }
    return mujoco.MjSpec.from_string(
        scene.read_text(encoding="utf-8"),
        include={name: (directory / name).read_bytes() for name in ("panda.xml", "scene.xml")},
        assets=assets,
    )


def load_panda() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = load_panda_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def require_position_servo(model: mujoco.MjModel, joint_name: str, actuator_name: str) -> None:
    joint = model.joint(joint_name)
    actuator = model.actuator(actuator_name)
    kp = actuator.gainprm[0]
    bias_offset, position_bias, velocity_bias = actuator.biasprm[:3]
    if not (
        joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE
        and actuator.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT
        and actuator.trnid[0] == joint.id
        and actuator.gear[0] == 1.0
        and actuator.dyntype[0] == mujoco.mjtDyn.mjDYN_NONE
        and actuator.gaintype[0] == mujoco.mjtGain.mjGAIN_FIXED
        and actuator.biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
        and kp > 0
        and bias_offset == 0
        and position_bias == -kp
        and velocity_bias <= 0
    ):
        raise ValueError(f"{actuator_name} must be a position servo for {joint_name} with a 1:1 transmission")


def simulate(
    target_rad: float = DEFAULT_TARGET_RAD, *, viewer: bool = False, tracking: bool = False
) -> list[Sample]:
    model, data = load_panda()
    require_position_servo(model, JOINT, ACTUATOR)
    joint = model.joint(JOINT)
    joint_position = data.joint(JOINT).qpos
    actuator = model.actuator(ACTUATOR)

    lower = max(joint.range[0], actuator.ctrlrange[0])
    upper = min(joint.range[1], actuator.ctrlrange[1])
    home_target = float(data.ctrl[actuator.id])
    if tracking:
        if not lower <= home_target - TRACKING_AMPLITUDE_RAD <= home_target + TRACKING_AMPLITUDE_RAD <= upper:
            raise ValueError(f"{JOINT} tracking trajectory exceeds [{lower}, {upper}] rad")
    elif not math.isfinite(target_rad) or not lower <= target_rad <= upper:
        raise ValueError(f"{JOINT} target must be finite and within [{lower}, {upper}] rad")

    timestep = model.opt.timestep
    step_at = round(STEP_TIME_S / timestep)
    duration = TRACKING_DURATION_S if tracking else DURATION_S
    steps = round(duration / timestep)
    viewer_speed = TRACKING_VIEWER_SPEED if tracking else VIEWER_SPEED
    samples = []
    window_context = nullcontext()
    if viewer:
        from mujoco.viewer import launch_passive

        # Keep Control-panel edits out of the scripted simulation.
        view_data = mujoco.MjData(model)
        mujoco.mj_copyData(view_data, model, data)
        window_context = launch_passive(model, view_data, show_left_ui=False, show_right_ui=False)

    with window_context as window:
        start = perf_counter()
        if window is not None:
            with window.lock():
                window.cam.lookat[:] = (0, 0, 0.52)
                window.cam.distance = 2.2
                window.cam.azimuth = 125
                window.cam.elevation = -22
            window.sync()
            start = perf_counter() + VIEWER_LEAD_IN_S
        for step in range(steps + 1):
            if tracking:
                cycle_time = (step * timestep) % TRACKING_PERIOD_S
                phase = math.tau * cycle_time / TRACKING_PERIOD_S
                data.ctrl[actuator.id] = home_target + TRACKING_AMPLITUDE_RAD * math.sin(phase)
            elif step == step_at:
                data.ctrl[actuator.id] = target_rad
            samples.append(Sample(float(data.time), float(data.ctrl[actuator.id]), float(joint_position[0])))
            if window is not None and window.is_running():
                sleep(max(0.0, start + data.time / viewer_speed - perf_counter()))
                with window.lock():
                    mujoco.mj_copyData(view_data, model, data)
                window.sync(state_only=True)
            if step < steps:
                mujoco.mj_step(model, data)

    if not all(math.isfinite(sample.angle_rad) for sample in samples) or not math.isclose(
        samples[-1].time_s, duration, rel_tol=0.0, abs_tol=timestep / 2
    ):
        raise RuntimeError("Panda simulation did not complete with finite joint angles")
    return samples


def save_results(
    samples: list[Sample], output_dir: Path, *, tracking: bool = False
) -> tuple[Path, Path]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    csv_path = output_dir / ("joint1_tracking.csv" if tracking else "joint_angle.csv")
    png_path = output_dir / ("figures/joint1_tracking.png" if tracking else "joint_angle.png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        if tracking:
            writer.writerow((*Sample._fields, "error_rad"))
            writer.writerows((*sample, sample.error_rad) for sample in samples)
        else:
            writer.writerow(Sample._fields)
            writer.writerows(samples)

    times, targets, angles = zip(*samples)
    figure = Figure(figsize=(8, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    if tracking:
        axes.plot(times, targets, linestyle="--", label="Target")
        title = "Panda joint1 position tracking"
    else:
        axes.step(times, targets, where="post", linestyle="--", label="Target")
        title = "Panda joint1 position step response"
    axes.plot(times, angles, label="Measured")
    axes.set(title=title, xlabel="Time (s)", ylabel="Joint angle (rad)")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.savefig(png_path, dpi=150)
    return csv_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--target", type=float, default=DEFAULT_TARGET_RAD, help="joint1 step target in radians")
    mode.add_argument("--tracking", action="store_true", help="run two cycles of sinusoidal joint1 tracking")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="CSV and PNG directory")
    parser.add_argument("--viewer", action="store_true", help="show the motion in the MuJoCo 3D viewer")
    args = parser.parse_args()
    try:
        samples = simulate(args.target, viewer=args.viewer, tracking=args.tracking)
    except ValueError as error:
        parser.error(str(error))
    csv_path, png_path = save_results(samples, args.output_dir, tracking=args.tracking)
    first, last = samples[0], samples[-1]
    print(f"MuJoCo {mujoco.__version__} | {JOINT} | {len(samples)} samples | {last.time_s:.3f} s")
    print(f"Angle: {first.angle_rad:.6f} -> {last.angle_rad:.6f} rad; target: {last.target_rad:.6f} rad")
    print(f"Final error: {abs(last.error_rad):.6f} rad")
    if args.tracking:
        angles = [sample.angle_rad for sample in samples]
        max_error = max(abs(sample.error_rad) for sample in samples)
        rmse = math.sqrt(sum(sample.error_rad**2 for sample in samples) / len(samples))
        print(f"Measured angle range: [{min(angles):.6f}, {max(angles):.6f}] rad")
        print(f"Tracking error: max abs {max_error:.6f} rad | RMS {rmse:.6f} rad")
    print(f"CSV: {csv_path.resolve()}\nPNG: {png_path.resolve()}")


if __name__ == "__main__":
    main()
