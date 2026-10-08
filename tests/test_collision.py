import os
import sys
import numpy as np
from spatialmath import SE3

# Allow tests/ to import files from AT 2/
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from safety import CollisionSafety
from workcell import make_arms

# Basic creation
def test_collision_safety_created():
    safety = CollisionSafety()

    assert safety is not None

# Self collision
def test_kawasaki_self_collision_runs():
    arm1, _, _ = make_arms()

    safety = CollisionSafety()

    result = safety.self_collision(arm1)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "reason" in result


def test_staubli_self_collision_runs():
    _, arm2, _ = make_arms()

    safety = CollisionSafety()

    result = safety.self_collision(arm2)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "reason" in result


def test_nachi_self_collision_runs():
    _, _, arm3 = make_arms()

    safety = CollisionSafety()
    result = safety.self_collision(arm3)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "reason" in result


# Robot-to-robot collision
def test_kawasaki_staubli_collision_check_runs():
    arm1, arm2, _ = make_arms()
    safety = CollisionSafety()

    result = safety.robot_robot(arm1, arm2)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "reason" in result


def test_staubli_nachi_collision_check_runs():
    _, arm2, arm3 = make_arms()

    safety = CollisionSafety()
    result = safety.robot_robot(arm2, arm3)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "reason" in result


def test_kawasaki_nachi_collision_check_runs():
    arm1, _, arm3 = make_arms()
    safety = CollisionSafety()

    result = safety.robot_robot(
        arm1,
        arm3,
    )

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "reason" in result

# Artificial dangerous condition
def test_close_robots_trigger_collision_warning():
    """
    Artificially place two robot bases almost on top
    of each other.

    The collision monitor should identify that their
    link geometry is dangerously close.
    """

    arm1, arm2, _ = make_arms()

    safety = CollisionSafety(robot_clearance=0.10)

    # Move Staubli almost onto Kawasaki
    arm2.robot.base = arm1.robot.base * SE3(0.02,0.0,0.0,)

    result = safety.robot_robot(arm1,arm2,)
    assert result["safe"] is False

# Obstacle collision
def test_sphere_obstacle_check_runs():
    arm1, _, _ = make_arms()

    safety = CollisionSafety()

    obstacle_centre = np.array([2.0,2.0,2.0,])
    result = safety.sphere_obstacle (arm1,sphere_center=obstacle_centre,sphere_radius=0.10,)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "reason" in result