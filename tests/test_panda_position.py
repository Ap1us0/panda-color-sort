import csv
import math
import subprocess
import sys

import mujoco
import numpy as np
import pytest

import panda_position
from panda_position import ROOT, load_panda, simulate


def test_loads_panda_with_joint1_position_servo():
    model, data = load_panda()
    joint = model.joint("joint1")
    actuator = model.actuator("actuator1")

    assert (model.nq, model.nv, model.nu) == (9, 9, 8)
    assert joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE
    assert actuator.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT
    assert actuator.trnid[0] == joint.id
    assert actuator.gainprm[0] > 0
    assert actuator.biasprm[1] == -actuator.gainprm[0]
    assert data.joint("joint1").qpos[0] == pytest.approx(0.0)


@pytest.mark.parametrize("target", [0.5, -0.3])
def test_joint1_tracks_step(target):
    model, _ = load_panda()
    timestep = model.opt.timestep
    samples = simulate(target)
    step_at = round(0.5 / timestep)

    assert len(samples) == round(3.0 / timestep) + 1
    assert samples[0].time_s == 0.0
    assert samples[step_at].time_s == pytest.approx(0.5)
    assert samples[-1].time_s == pytest.approx(3.0)
    assert all(after.time_s - before.time_s == pytest.approx(timestep) for before, after in zip(samples, samples[1:]))
    assert all(math.isfinite(sample.angle_rad) for sample in samples)
    assert all(sample.target_rad == 0.0 for sample in samples[:step_at])
    assert all(sample.target_rad == target for sample in samples[step_at:])
    assert samples[-1].angle_rad == pytest.approx(target, abs=0.02)


@pytest.mark.parametrize("target", [4.0, -4.0, float("nan"), float("inf")])
def test_rejects_invalid_target(target):
    with pytest.raises(ValueError, match="joint1 target must be finite and within"):
        simulate(target)


def test_cli_exports_csv_and_png_from_another_working_directory(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "panda_position.py"), "--output-dir", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Final error:" in result.stdout
    with (output / "joint_angle.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["time_s", "target_rad", "angle_rad"]
        rows = list(reader)
    assert float(rows[-1]["time_s"]) == pytest.approx(3.0)
    assert float(rows[-1]["angle_rad"]) == pytest.approx(0.5, abs=0.02)
    with (output / "joint_angle.png").open("rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"


def test_tracking_rejects_velocity_actuator(monkeypatch):
    model, data = load_panda()
    model.actuator("actuator1").biasprm[1] = 0.0
    monkeypatch.setattr(panda_position, "load_panda", lambda: (model, data))
    with pytest.raises(ValueError, match="must be a position servo"):
        simulate(tracking=True)


def test_tracking_runs_two_cycles_with_finite_dynamics(monkeypatch):
    model, home = load_panda()
    timestep = model.opt.timestep
    home_controls = home.ctrl[1:].copy()
    home_arm_pose = home.qpos[1:7].copy()
    step_physics = mujoco.mj_step
    steps = 0

    def checked_step(model, data):
        nonlocal steps
        assert np.isfinite(data.ctrl).all()
        assert -0.8 <= data.ctrl[0] <= 0.8
        np.testing.assert_array_equal(data.ctrl[1:], home_controls)
        step_physics(model, data)
        steps += 1
        assert np.isfinite(data.qpos).all()
        assert np.isfinite(data.qvel).all()
        assert not any(warning.number for warning in data.warning)
        assert abs(data.qvel[0]) < 2.0  # rad/s for this trajectory
        assert np.max(np.abs(data.qpos[1:7] - home_arm_pose)) < 0.03  # rad

    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    samples = simulate(tracking=True)

    assert steps >= 1000
    assert steps == round(8.0 / timestep)
    assert len(samples) == steps + 1
    assert samples[-1].time_s == pytest.approx(8.0)
    assert max(abs(sample.error_rad) for sample in samples) < 0.20
    for time_s, expected_target in [(1, 0.8), (3, -0.8), (5, 0.8), (7, -0.8)]:
        sample = samples[round(time_s / timestep)]
        assert sample.target_rad == pytest.approx(expected_target)
        assert sample.angle_rad == pytest.approx(expected_target, abs=0.05)


def test_tracking_cli_exports_error_and_figure(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "panda_position.py"), "--tracking", "--output-dir", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Tracking error:" in result.stdout
    with (output / "joint1_tracking.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["time_s", "target_rad", "angle_rad", "error_rad"]
        rows = list(reader)
    assert len(rows) == 4001
    assert all(math.isfinite(float(value)) for row in rows for value in row.values())
    for row in rows:
        assert float(row["error_rad"]) == pytest.approx(float(row["target_rad"]) - float(row["angle_rad"]))
    with (output / "figures" / "joint1_tracking.png").open("rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
