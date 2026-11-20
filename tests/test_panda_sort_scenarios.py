from collections import Counter
from itertools import combinations
import json

import mujoco
import numpy as np

from panda_color_sort import SLOTS
from panda_pick_place import reset_scene
from panda_sort_evaluate import make_cases, rates
from panda_sort_scenarios import (
    MIN_PAIR_DISTANCE_M, MIN_SLOT_DISTANCE_M, classify_failure, generate_spawns, prepare_scenario,
    ranked_assignments, run_scenario,
)


def test_seeded_scenario_spawns_fit_the_table_and_slot_capacity():
    scene, spawns = prepare_scenario(3, 7301)
    assert spawns == generate_spawns(3, 7301)
    assert spawns != generate_spawns(3, 7302)
    counts = Counter(cube.color for cube in spawns)
    assert all(0 < count <= len(SLOTS[color]) for color, count in counts.items())
    table = scene.model.geom("table")
    data = reset_scene(scene.model)
    assert not any(contact.dist < 0 for contact in data.contact)
    for a, b in combinations(spawns, 2):
        assert np.hypot(a.x_m - b.x_m, a.y_m - b.y_m) >= MIN_PAIR_DISTANCE_M
    for item, cube in zip(scene.items, spawns):
        xy = np.array([cube.x_m, cube.y_m])
        radius = np.linalg.norm(scene.model.geom_size[item.task.cube, :2])
        assert np.all(np.abs(xy - table.pos[:2]) + radius <= table.size[:2])
        for slots in SLOTS.values():
            assert all(np.linalg.norm(xy - [x, y]) >= MIN_SLOT_DISTANCE_M for _, x, y in slots)


def test_ranked_assignments_recheck_physically_occupied_slots():
    scene, _ = prepare_scenario(3, 7301)
    data = reset_scene(scene.model)
    pending = {item.name: item for item in scene.items}
    assert {item.slot for item in ranked_assignments(scene, data, pending, {})} == {"red_1", "red_2", "blue_1"}
    occupied = {"red_1": "earlier_cube"}
    assert all(item.slot != "red_1" for item in ranked_assignments(scene, data, pending, occupied))


def test_simulation_cap_safely_preserves_seed_and_does_not_step_past_limit(monkeypatch):
    scene, _ = prepare_scenario(2, 7300)
    previous = reset_scene(scene.model).qpos.copy()
    step = mujoco.mj_step
    count = 0

    def checked_step(model, data):
        nonlocal count
        np.testing.assert_array_equal(data.qpos, previous)
        step(model, data)
        previous[:] = data.qpos
        count += 1

    monkeypatch.setattr(mujoco, "mj_step", checked_step)
    result = run_scenario(scene, 7300, max_steps=23)
    assert count == result["scene"]["sim_steps"] == 23
    assert result["scene"]["seed"] == 7300 and not result["scene"]["success"]
    assert result["scene"]["failure_type"] == "timeout"
    assert len(result["objects"]) == 2 and not any(obj["correct"] for obj in result["objects"])


def test_evaluation_counts_partial_results_and_classifies_failures():
    assert make_cases(5, 8) == [(8, 2), (9, 3), (10, 2), (11, 3), (12, 2)]
    rows = [{"cube_count": 2, "correct_cubes": 2, "success": True},
            {"cube_count": 3, "correct_cubes": 1, "success": False}]
    assert rates(rows) == (0.6, 0.5, 3, 5)
    assert classify_failure("PLAN", "path: collision finger / obstacle") == "collision"
    assert classify_failure("PLAN", "IK did not converge within joint limits") == "ik_failure"
    assert classify_failure("GRASP", "bilateral load-bearing grasp did not last") == "grasp_failure"
    assert classify_failure("TRANSPORT", "lost load-bearing contact") == "dropped"


def test_adaptive_success_replay_is_json_serializable():
    scene, _ = prepare_scenario(2, 7300)
    result = json.loads(json.dumps(run_scenario(scene, 7300)))
    assert result["scene"]["success"] is True
    assert result["scene"]["correct_cubes"] == 2
    assert all(obj["correct"] is True for obj in result["objects"])
