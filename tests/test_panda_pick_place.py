import mujoco
import numpy as np
import pytest

from panda_pick_place import (
    GOAL_RADIUS_M,
    MAX_CONTACT_DEPTH_M,
    MIN_HOLD_S,
    MIN_LIFT_M,
    Gripper,
    State,
    load_scene,
    prepare_task,
    reset_scene,
    run_trial,
)


def step(model, data, duration):
    for _ in range(round(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def test_cube_falls_and_rests_on_table_with_original_gripper():
    model = load_scene()
    data = reset_scene(model)
    gripper = Gripper.from_model(model)
    cube, table = model.geom("cube").id, model.geom("table").id
    top = model.geom_pos[table, 2] + model.geom_size[table, 2]
    rest_z = top + model.geom_size[cube, 2]
    assert model.joint("cube_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body("cube").mass[0] == pytest.approx(0.075)
    assert all(model.eq_type != mujoco.mjtEq.mjEQ_WELD)
    assert gripper.opened == 255 and gripper.closed == 0
    assert data.geom_xpos[cube, 2] - rest_z > 0.005

    step(model, data, 1.0)
    assert data.geom_xpos[cube, 2] == pytest.approx(rest_z, abs=0.001)
    assert any(set(contact.geom) == {cube, table} and contact.efc_address >= 0
               for contact in data.contact)
    settled = data.geom_xpos[cube].copy()
    step(model, data, 0.5)
    np.testing.assert_allclose(data.geom_xpos[cube], settled, atol=1e-5)


def test_stock_gripper_closes_reopens_and_holds_with_dynamics():
    model = load_scene()
    data = reset_scene(model)
    gripper = Gripper.from_model(model)
    fingers = list(gripper.fingers)
    np.testing.assert_allclose(data.qpos[fingers], 0.04)
    for opening, expected in ((0.0, 0.0), (0.5, 0.02), (1.0, 0.04)):
        gripper.command(data, opening)
        step(model, data, 2.5)
        np.testing.assert_allclose(data.qpos[fingers], expected, atol=0.001)
        held = data.qpos[fingers].copy()
        step(model, data, 0.3)
        np.testing.assert_allclose(data.qpos[fingers], held, atol=0.0002)
    assert not any(warning.number for warning in data.warning)


@pytest.fixture(scope="module")
def task():
    return prepare_task()


def test_pick_place_runs_every_state_using_only_physics(task, monkeypatch):
    physics_step = mujoco.mj_step
    previous = reset_scene(task.model).qpos.copy()
    steps = 0
    states = []

    def checked_step(model, data):
        nonlocal steps
        np.testing.assert_array_equal(data.qpos, previous)
        assert task.gripper.closed <= data.ctrl[task.gripper.actuator] <= task.gripper.opened
        physics_step(model, data)
        previous[:] = data.qpos
        steps += 1

    def observe(state, data, contacts):
        if not states or state != states[-1]:
            states.append(state)
        if state in (State.HOLD, State.TRANSPORT):
            assert data.geom_xpos[task.cube, 2] >= task.rest_z + MIN_LIFT_M
            assert contacts.gripping and not contacts.table_count
        if state == State.RETREAT:
            assert contacts.supported and not (contacts.left_count or contacts.right_count)

    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    result = run_trial(task, observe)
    assert result.success, result.reason
    assert states == list(State)
    assert steps * task.model.opt.timestep == pytest.approx(result.time_s)
    assert result.max_lift_m >= MIN_LIFT_M
    assert result.hold_s >= MIN_HOLD_S and result.stable_s >= 1.0
    assert result.final_xy_error_m < GOAL_RADIUS_M
    assert max(result.max_finger_depth_m, result.max_table_depth_m) < MAX_CONTACT_DEPTH_M


def test_grasp_fails_when_no_finger_contact(task, monkeypatch):
    command = Gripper.command
    monkeypatch.setattr(Gripper, "command", lambda self, data, opening: command(self, data, 1.0))
    result = run_trial(task)
    assert not result.success
    assert result.state == State.GRASP.name
    assert "bilateral" in result.reason
    assert result.max_lift_m == 0.0
