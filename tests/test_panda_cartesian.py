import csv

import mujoco
import numpy as np
import pytest

from panda_cartesian import (
    DEMO_ROTATION_DEG,
    FINAL_ERROR_LIMIT_M,
    JOINT_MARGIN_RAD,
    MAX_REFERENCE_ACCEL,
    MAX_REFERENCE_SPEED,
    MAX_TRACKING_ACCEL,
    MAX_TRACKING_SPEED,
    ORIENTATION_ERROR_LIMIT_DEG,
    ORIENTATION_WEIGHT,
    SINGULAR_VALUE_THRESHOLD,
    InverseKinematics,
    damped_step,
    interpolate_orientation,
    load_panda,
    plan_trajectory,
    read_arm,
    rotation_error,
    save_results,
    tcp_position,
    tcp_pose,
    track_trajectory,
    tracking_errors,
)
from panda_position import load_panda_spec


def test_ik_rejects_an_unreachable_target():
    model, home = load_panda()
    arm = read_arm(model)
    ik = InverseKinematics(model, home, arm)
    with pytest.raises(ValueError, match="unreachable.*XYZ residual"):
        ik.solve(np.array([2.0, 0.0, 1.2]), home.qpos[arm.qpos])


def test_preflight_rejects_a_colliding_home_pose():
    spec = load_panda_spec()
    spec.worldbody.add_geom(
        name="collision_fixture", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.5545, 0, 0.61], size=[0.08, 0, 0],
    )
    model = spec.compile()
    home = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, home, model.key("home").id)
    mujoco.mj_forward(model, home)
    arm = read_arm(model)
    with pytest.raises(RuntimeError, match=r"path at 0\.000 s \(home\): collision"):
        plan_trajectory(model, home, arm)


def test_damped_ik_handles_a_near_singular_panda_pose():
    model, home = load_panda()
    arm = read_arm(model)
    ik = InverseKinematics(model, home, arm)
    candidate = mujoco.MjData(model)
    mujoco.mj_copyData(candidate, model, home)
    angles = np.array([0.375, -1.088, -0.379, -0.125, 1.277, 1.893, -0.101])
    candidate.qpos[arm.qpos] = angles
    mujoco.mj_fwdPosition(model, candidate)
    xyz, orientation = tcp_pose(candidate, arm)
    jacp = np.empty((3, model.nv))
    jacr = np.empty_like(jacp)
    mujoco.mj_jac(model, candidate, jacp, jacr, xyz, arm.hand)
    jacobian = np.vstack((jacp[:, arm.dofs], ORIENTATION_WEIGHT * jacr[:, arm.dofs]))
    u, singular_values, _ = np.linalg.svd(jacobian, full_matrices=False)
    assert singular_values[-1] < SINGULAR_VALUE_THRESHOLD
    assert np.linalg.norm(damped_step(jacobian, u[:, -1])) < 0.8 / singular_values[-1]

    seed = angles + np.array([0.01, -0.02, 0.01, 0.005, 0.01, 0.01, 0.01])
    candidate.qpos[arm.qpos] = ik.solve(xyz, seed, orientation=orientation)
    mujoco.mj_fwdPosition(model, candidate)
    actual_xyz, actual_orientation = tcp_pose(candidate, arm)
    np.testing.assert_allclose(actual_xyz, xyz, atol=1e-7)
    np.testing.assert_allclose(rotation_error(orientation, actual_orientation), 0.0, atol=1e-7)


def test_rotation_interpolation_takes_the_short_arc_across_180_degrees():
    start = np.empty(4)
    end = np.empty(4)
    mujoco.mju_axisAngle2Quat(start, [0, 0, 1], np.deg2rad(170))
    mujoco.mju_axisAngle2Quat(end, [0, 0, 1], np.deg2rad(-170))
    middle, finish = interpolate_orientation(start, end, np.array([0.5, 1.0]))
    np.testing.assert_allclose(np.rad2deg(rotation_error(middle, start)), [0, 0, 10], atol=1e-10)
    np.testing.assert_allclose(rotation_error(end, finish), 0.0, atol=1e-12)
    np.testing.assert_allclose(rotation_error(-start, start), 0.0, atol=1e-12)


def test_cartesian_path_tracks_through_dynamics_and_exports(monkeypatch, tmp_path):
    model, data = load_panda()
    arm = read_arm(model)
    plan = plan_trajectory(model, data, arm)
    dt = model.opt.timestep
    assert len(arm.actuators) == 7
    assert plan.peak_speed <= MAX_REFERENCE_SPEED
    assert plan.peak_accel <= MAX_REFERENCE_ACCEL
    assert np.all(plan.angles >= arm.joint_range[:, 0] + JOINT_MARGIN_RAD)
    assert np.all(plan.angles <= arm.joint_range[:, 1] - JOINT_MARGIN_RAD)
    np.testing.assert_allclose(plan.waypoints[0], tcp_position(data, arm), atol=1e-12)
    delta_xyz = np.diff(plan.waypoints, axis=0)
    assert delta_xyz[0, 2] == pytest.approx(0)
    np.testing.assert_allclose(delta_xyz[1:3, :2], 0, atol=1e-12)
    assert delta_xyz[1, 2] < 0 < delta_xyz[2, 2]

    step_physics = mujoco.mj_step
    previous_qpos = data.qpos.copy()
    grip = data.ctrl[model.actuator("actuator8").id]
    steps = 0

    def checked_step(step_model, step_data):
        nonlocal steps
        assert step_data is data
        np.testing.assert_array_equal(step_data.qpos, previous_qpos)
        assert step_data.actuator("actuator8").ctrl[0] == grip
        assert np.all(step_data.ctrl[arm.actuators] >= arm.ctrl_range[:, 0])
        assert np.all(step_data.ctrl[arm.actuators] <= arm.ctrl_range[:, 1])
        step_physics(step_model, step_data)
        previous_qpos[:] = step_data.qpos
        steps += 1

    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    run = track_trajectory(model, data, arm, plan)
    assert steps == len(run.times) - 1 >= 1000
    assert run.times[-1] == pytest.approx(10.0)
    assert np.isfinite(run.angles).all() and np.isfinite(run.velocities).all() and np.isfinite(run.xyz).all()
    assert np.all(np.ptp(run.angles, axis=0) > 0.02)
    assert np.all(np.ptp(run.xyz, axis=0) > 0.05)
    assert np.max(np.abs(run.velocities)) <= MAX_TRACKING_SPEED
    assert np.max(np.abs(np.diff(run.velocities, axis=0))) / dt <= MAX_TRACKING_ACCEL
    assert np.all(run.torques >= arm.torque_range[:, 0] - 1e-9)
    assert np.all(run.torques <= arm.torque_range[:, 1] + 1e-9)
    xyz_error, rotation_error_deg = tracking_errors(plan, run)
    error = np.linalg.norm(xyz_error, axis=1)
    angle_error = np.linalg.norm(rotation_error_deg, axis=1)
    assert error.max() < FINAL_ERROR_LIMIT_M
    assert angle_error.max() < ORIENTATION_ERROR_LIMIT_DEG
    np.testing.assert_allclose(np.linalg.norm(run.quaternions, axis=1), 1.0, atol=1e-12)
    held = slice(None, plan.waypoint_steps[-2] + 1)
    np.testing.assert_allclose(plan.quaternions[held] - plan.quaternions[0], 0.0, atol=1e-12)
    change_deg = np.rad2deg(np.linalg.norm(rotation_error(run.quaternions[-1], run.quaternions[0])))
    assert change_deg == pytest.approx(DEMO_ROTATION_DEG, abs=0.1)

    csv_path, *figures = save_results(plan, run, tmp_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames[:7] == [
            "time_s", "target_x_m", "target_y_m", "target_z_m", "actual_x_m", "actual_y_m", "actual_z_m"
        ]
        rows = list(reader)
    assert len(rows) == steps + 1
    assert float(rows[-1]["error_norm_m"]) == pytest.approx(error[-1])
    assert float(rows[-1]["orientation_error_deg"]) == pytest.approx(angle_error[-1])
    assert float(rows[-1]["actual_quat_w"]) == pytest.approx(run.quaternions[-1, 0])
    for figure in figures:
        assert figure.parent == tmp_path / "figures"
        with figure.open("rb") as stream:
            assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
