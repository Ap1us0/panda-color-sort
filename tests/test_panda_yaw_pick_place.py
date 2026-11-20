import mujoco
import numpy as np
import pytest

import panda_pick_place as pp


def test_seeded_xy_and_yaw_are_reproducible():
    poses = pp.sample_pickup_poses(30, pp.DEFAULT_SEED)
    np.testing.assert_array_equal(poses, pp.sample_pickup_poses(30, pp.DEFAULT_SEED))
    assert not np.array_equal(poses, pp.sample_pickup_poses(30, pp.DEFAULT_SEED + 1))
    assert np.all(poses[:, :2] >= pp.SPAWN_MIN_XY)
    assert np.all(poses[:, :2] <= pp.SPAWN_MAX_XY)
    assert np.all((-np.pi <= poses[:, 2]) & (poses[:, 2] < np.pi))
    assert np.unique(poses, axis=0).shape == (30, 3)


def test_grasp_alignment_chooses_the_nearest_equivalent_face():
    jaw_yaw = np.deg2rad(8)
    for cube_yaw_deg in (-179, -135, -46, 45, 170):
        cube_yaw = np.deg2rad(cube_yaw_deg)
        offset = pp.square_yaw_offset(cube_yaw, jaw_yaw)
        assert abs(offset) <= np.pi / 4
        assert pp.square_yaw_offset(cube_yaw, jaw_yaw + offset) == pytest.approx(0.0, abs=1e-12)
        assert pp.square_yaw_offset(cube_yaw + np.pi / 2, jaw_yaw) == pytest.approx(offset)


@pytest.fixture(scope="module")
def task():
    return pp.prepare_task()


def test_rotated_cube_reset_changes_only_its_freejoint_and_checks_footprint(task):
    model = task.model
    xy = np.array([0.43, -0.16])
    yaw = np.deg2rad(-136)
    home = pp.reset_scene(model)
    data = pp.reset_scene(model, xy, cube_yaw=yaw)
    start = int(model.joint("cube_free").qposadr[0])
    expected = home.qpos.copy()
    expected[start:start + 2] = xy
    expected[start + 3:start + 7] = pp.world_z_quaternion(yaw)
    np.testing.assert_array_equal(data.qpos, expected)
    assert pp.planar_yaw(data.geom_xmat[task.cube]) == pytest.approx(yaw)
    assert np.all(data.qvel == 0)
    np.testing.assert_array_equal(pp.reset_scene(model).qpos, home.qpos)

    edge = np.array([0.725, 0.0])
    pp.reset_scene(model, edge)
    with pytest.raises(ValueError, match="footprint"):
        pp.reset_scene(model, edge, cube_yaw=np.pi / 4)


def test_rotated_grasp_tracks_yaw_with_real_contact_and_fixed_goal(task, monkeypatch):
    *xy, yaw = pp.validation_poses()[0]
    previous = pp.reset_scene(task.model, xy, cube_yaw=yaw).qpos.copy()
    physics_step = mujoco.mj_step

    def checked_step(model, data):
        np.testing.assert_array_equal(data.qpos, previous)
        physics_step(model, data)
        previous[:] = data.qpos

    def observe(state, data, contacts):
        if state is pp.State.HOLD:
            assert contacts.gripping and not contacts.table_count

    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    result = pp.run_trial(task, observe, cube_xy=xy, cube_yaw=yaw)
    assert result.success, result.reason
    assert result.initial_yaw_deg == pytest.approx(np.rad2deg(yaw))
    assert abs(result.grasp_yaw_offset_deg) > 30.0
    assert result.grasp_alignment_error_deg < 0.2
    assert result.final_xy_error_m < pp.GOAL_RADIUS_M
    assert result.final_square_yaw_error_deg < pp.GOAL_YAW_TOL_DEG
