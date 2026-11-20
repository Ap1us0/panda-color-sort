import mujoco
import numpy as np
import pytest

import panda_pick_place as pick_place
from panda_color_sort import final_slot_check, prepare_sort, run_sort
from panda_pick_place import DEFAULT_SEED, State, reset_scene


@pytest.mark.parametrize("count", (2, 3))
def test_color_sort_uses_unique_matching_slots(count):
    scene = prepare_sort(count, DEFAULT_SEED)
    assert len(scene.items) == count
    assert len({item.slot for item in scene.items}) == count
    assert {item.color for item in scene.items} == {"red", "blue"}
    assert sum(item.color == "red" for item in scene.items) == count - 1
    for item in scene.items:
        assert item.slot.startswith(f"{item.color}_")
        np.testing.assert_array_equal(
            scene.model.geom_rgba[item.task.cube], scene.model.site(item.slot).rgba,
        )
    assert all(scene.model.eq_type != mujoco.mjtEq.mjEQ_WELD)


def test_three_cubes_are_sorted_in_one_continuous_physics_run(monkeypatch):
    scene = prepare_sort(3, DEFAULT_SEED)
    previous = reset_scene(scene.model).qpos.copy()
    step = mujoco.mj_step
    steps = 0
    held = set()
    live_states = set()

    def checked_step(model, data):
        nonlocal steps
        np.testing.assert_array_equal(data.qpos, previous)
        step(model, data)
        previous[:] = data.qpos
        steps += 1

    def observe(item, state, data, contacts):
        live_states.add(id(data))
        if state is State.HOLD:
            assert contacts.gripping and not contacts.table_count
            held.add(item.name)

    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    data, results = run_sort(scene, observe)
    assert len(live_states) == 1
    assert [result.success for result in results] == [True] * 3
    assert held == {item.name for item in scene.items}
    assert all(final_slot_check(item, data)[2] for item in scene.items)
    assert steps == 31725
    assert data.time == pytest.approx(63.45)


def test_continued_pick_retimes_an_over_limit_plan(monkeypatch):
    task = prepare_sort(2, 7).items[0].task
    home = reset_scene(task.model)
    pickup = home.geom_xpos[task.cube].copy()
    timing = []
    retimed = object()

    def plan(model, data, arm, targets):
        timing.append(np.array([target.duration_s for target in targets]))
        if len(timing) == 1:
            raise ValueError("Joint reference exceeds velocity or acceleration limits: 0.960 rad/s")
        return retimed

    monkeypatch.setattr(pick_place, "plan_trajectory", plan)
    result = pick_place.plan_pickup(task, pickup, start_angles=home.qpos[task.arm.qpos])
    assert result is retimed
    np.testing.assert_allclose(timing[1], 1.5 * timing[0])
