import csv

import mujoco
import numpy as np
import pytest

import panda_pick_place as pick_place


def test_seeded_spawns_are_reproducible_and_inside_the_sampling_region():
    positions = pick_place.sample_pickups(30, pick_place.DEFAULT_SEED)
    np.testing.assert_array_equal(positions, pick_place.sample_pickups(30, pick_place.DEFAULT_SEED))
    assert not np.array_equal(positions, pick_place.sample_pickups(30, pick_place.DEFAULT_SEED + 1))
    assert np.unique(positions, axis=0).shape == (30, 2)
    assert np.all(positions >= pick_place.SPAWN_MIN_XY)
    assert np.all(positions <= pick_place.SPAWN_MAX_XY)


@pytest.fixture(scope="module")
def task():
    return pick_place.prepare_task()


def test_random_reset_changes_only_cube_xy_and_rejects_off_table(task):
    model = task.model
    previous = pick_place.reset_scene(model)
    xy = pick_place.validation_pickups()[-1]
    data = pick_place.reset_scene(model, xy)
    cube_qpos = int(model.joint("cube_free").qposadr[0])
    expected = previous.qpos.copy()
    expected[cube_qpos:cube_qpos + 2] = xy
    np.testing.assert_array_equal(data.qpos, expected)
    np.testing.assert_array_equal(data.geom_xpos[task.cube, :2], xy)
    np.testing.assert_array_equal(pick_place.reset_scene(model).qpos, previous.qpos)
    assert data.time == 0.0 and not np.any(data.qvel)
    with pytest.raises(ValueError, match="footprint"):
        pick_place.reset_scene(model, np.array([0.75, 0.0]))


def test_nondefault_pickup_is_planned_from_settled_pose_and_uses_physics(task, monkeypatch):
    xy = pick_place.validation_pickups()[1]
    previous = pick_place.reset_scene(task.model, xy).qpos.copy()
    physics_step = mujoco.mj_step
    plan_pickup = pick_place.plan_pickup
    planned = []

    def checked_step(model, data):
        np.testing.assert_array_equal(data.qpos, previous)
        physics_step(model, data)
        previous[:] = data.qpos

    def inspect_plan(run_task, pickup):
        planned.append(pickup.copy())
        return plan_pickup(run_task, pickup)

    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    monkeypatch.setattr(pick_place, "plan_pickup", inspect_plan)
    result = pick_place.run_trial(task, cube_xy=xy)
    assert result.success, result.reason
    np.testing.assert_array_equal([result.initial_x_m, result.initial_y_m], xy)
    assert result.initial_z_m > result.settled_z_m
    assert len(planned) == 1
    assert planned[0] == pytest.approx([result.settled_x_m, result.settled_y_m, result.settled_z_m])
    assert result.max_lift_m >= pick_place.MIN_LIFT_M and result.hold_s >= pick_place.MIN_HOLD_S
    assert result.final_xy_error_m < pick_place.GOAL_RADIUS_M


def test_batch_counts_plan_failures_without_replacing_positions(task, monkeypatch, tmp_path):
    seed = 123
    positions = pick_place.sample_pickups(2, seed)

    def reject_plan(*args):
        raise ValueError("test IK failure")

    monkeypatch.setattr(pick_place, "plan_pickup", reject_plan)
    results = pick_place.run_batch(task, len(positions), tmp_path, positions=positions, seed=seed)
    assert len(results) == 2 and all(not result.success for result in results)
    with (tmp_path / "trials.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["trial"] for row in rows] == ["1", "2"]
    for row, xy in zip(rows, positions):
        assert row["seed"] == str(seed)
        assert row["state"] == "PLAN" and row["reason"] == "test IK failure"
        np.testing.assert_array_equal([float(row["initial_x_m"]), float(row["initial_y_m"])], xy)
