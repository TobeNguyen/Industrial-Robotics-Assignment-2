import os
import sys
import numpy as np
import pytest
from spatialmath import SE3

# Allow tests/ to import files from AT 2/
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from safety import (
    CollisionSafety,
    SingularitySafety,
    RobotSafetyController,
    ROBOT_VELOCITY_LIMITS,
    _get_robot,
    _point_segment_distance,
    _segment_segment_distance,
    _distance_scale,
    _valid_links
)
from workcell import make_arms
from kawasaki_rs007n_task import KawasakiPickPlace
from staubli_tx2_60_task import StaubliShapeDraw
from nachi_mz07_task import NachiShapeFill

# Fake robots (no hardware needed)
class FakeTransform:
    def __init__(self, point):
        self.t = np.array(point, dtype=float)


class FakeRobot:
    """
    Simple stand-in for a DHRobot3D.
    Lets safety.py be tested with known geometry and a known Jacobian.
    """
    def __init__(self, points=None, jacobian=None):
        self.q = np.zeros(6)
        self.points = points
        self.jacobian = jacobian

    def fkine_all(self, q=None):
        return [FakeTransform(p) for p in self.points]

    def jacob0(self, q=None):
        return self.jacobian


class FakeArm:
    """
    Wrapper class with a .robot attribute (like KawasakiPickPlace).
    """
    def __init__(self, robot):
        self.robot = robot


def vertical_robot(x):
    # One vertical link at position x
    return FakeRobot(points=[[x, 0, 0], [x, 0, 1]])


def straight_robot():
    # Fully stretched robot, no self-collision
    return FakeRobot(points=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])


def folded_robot(gap):
    # Last link ends 'gap' metres away from the first link
    return FakeRobot(points=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, gap, 0]])


def zero_length_robot():
    # Link 1 has zero length (two frames at the same position)
    return FakeRobot(points=[[0, 0, 0], [0, 0, 0.5], [0, 0, 0.5], [0, 0, 1]])


def jacobian_robot(J):
    return FakeRobot(points=[[0, 0, 0], [0, 0, 1]], jacobian=np.asarray(J, dtype=float))

# Helper safety results
def safe_collision():
    return {
        "safe": True
    }


def unsafe_collision():
    return {
        "safe": False
    }


def slow_collision(scale):
    return {
        "safe": True,
        "scale": scale
    }


def safe_singularity():
    return {
        "safe": True,
        "level": "SAFE"
    }


def warning_singularity():
    return {
        "safe": True,
        "level": "WARNING"
    }


def stop_singularity():
    return {
        "safe": False,
        "level": "STOP"
    }

# ======================================================================
# GEOMETRY HELPERS
# ======================================================================

# GET ROBOT
def test_get_robot_returns_inner_robot_from_wrapper():
    robot = vertical_robot(0.0)
    arm = FakeArm(robot)

    assert _get_robot(arm) is robot


def test_get_robot_returns_robot_unchanged():
    robot = vertical_robot(0.0)

    assert _get_robot(robot) is robot

# POINT TO SEGMENT
def test_point_segment_distance_perpendicular():
    distance = _point_segment_distance([0.5, 1.0, 0.0], [0, 0, 0], [1, 0, 0])

    assert distance == pytest.approx(1.0)


def test_point_segment_distance_beyond_end_uses_endpoint():
    distance = _point_segment_distance([2.0, 0.0, 0.0], [0, 0, 0], [1, 0, 0])

    assert distance == pytest.approx(1.0)


def test_point_segment_distance_zero_length_segment():
    distance = _point_segment_distance([0.0, 3.0, 4.0], [0, 0, 0], [0, 0, 0])

    assert distance == pytest.approx(5.0)

# SEGMENT TO SEGMENT
def test_segment_distance_parallel_segments():
    distance = _segment_segment_distance([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0])

    assert distance == pytest.approx(1.0)


def test_segment_distance_parallel_offset_segments():
    distance = _segment_segment_distance([0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0])

    assert distance == pytest.approx(np.sqrt(2.0))


def test_segment_distance_collinear_separated_segments():
    distance = _segment_segment_distance([0, 0, 0], [1, 0, 0], [3, 0, 0], [4, 0, 0])

    assert distance == pytest.approx(2.0)


def test_segment_distance_skew_segments():
    distance = _segment_segment_distance([-1, 0, 0], [1, 0, 0], [0, -1, 1], [0, 1, 1])

    assert distance == pytest.approx(1.0)


def test_segment_distance_intersecting_segments_is_zero():
    distance = _segment_segment_distance([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0])

    assert distance == pytest.approx(0.0, abs=1e-9)


def test_segment_distance_zero_length_segment():
    distance = _segment_segment_distance([0, 0, 1], [0, 0, 1], [0, 0, 0], [1, 0, 0])

    assert distance == pytest.approx(1.0)


def test_segment_distance_both_zero_length():
    distance = _segment_segment_distance([0, 0, 0], [0, 0, 0], [3, 4, 0], [3, 4, 0])

    assert distance == pytest.approx(5.0)


def test_segment_distance_is_symmetric():
    p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 0.5])
    q1, q2 = np.array([2.0, -1.0, 1.0]), np.array([0.5, 3.0, -0.5])

    d1 = _segment_segment_distance(p1, p2, q1, q2)
    d2 = _segment_segment_distance(q1, q2, p1, p2)

    assert d1 == pytest.approx(d2)


def test_segment_distance_matches_brute_force():
    """
    Compare against a brute force search over many points on both segments.
    The result must never be larger than the true distance (otherwise a collision could be missed).
    """
    rng = np.random.default_rng(0)
    s = np.linspace(0.0, 1.0, 300)[:, None]

    for _ in range(50):
        p1, p2, q1, q2 = rng.normal(size=(4, 3))

        points_p = p1 + s * (p2 - p1)
        points_q = q1 + s * (q2 - q1)
        brute = np.min(np.linalg.norm(points_p[:, None] - points_q[None], axis=2))

        distance = _segment_segment_distance(p1, p2, q1, q2)

        assert distance <= brute + 1e-9
        assert brute - distance < 0.05

# VALID LINKS
def test_valid_links_ignores_zero_length_links():
    points = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 0.5], [0, 0, 1]], dtype=float)

    assert _valid_links(points) == [0, 2]


def test_valid_links_keeps_all_normal_links():
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=float)

    assert _valid_links(points) == [0, 1]

# DISTANCE SCALE
def test_distance_scale_stops_inside_clearance():
    assert _distance_scale(0.05, 0.10, 2.0) == 0.0


def test_distance_scale_stops_at_clearance():
    assert _distance_scale(0.10, 0.10, 2.0) == 0.0


def test_distance_scale_full_speed_far_away():
    assert _distance_scale(0.50, 0.10, 2.0) == 1.0


def test_distance_scale_full_speed_at_slow_distance():
    assert _distance_scale(0.20, 0.10, 2.0) == 1.0


def test_distance_scale_is_linear_between_limits():
    assert _distance_scale(0.15, 0.10, 2.0) == pytest.approx(0.5)


def test_distance_scale_infinite_distance():
    assert _distance_scale(np.inf, 0.10, 2.0) == 1.0

# ======================================================================
# COLLISION SAFETY
# ======================================================================

# Basic creation
def test_collision_safety_created():
    safety = CollisionSafety()

    assert safety is not None


def test_collision_safety_default_values():
    safety = CollisionSafety()

    assert safety.robot_clearance == 0.10
    assert safety.self_clearance == 0.06
    assert safety.obstacle_clearance == 0.05
    assert safety.slow_factor == 2.0

# Self collision
def test_kawasaki_self_collision_runs():
    arm1, _, _ = make_arms()

    safety = CollisionSafety()

    result = safety.self_collision(arm1)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "scale" in result
    assert "reason" in result


def test_staubli_self_collision_runs():
    _, arm2, _ = make_arms()

    safety = CollisionSafety()

    result = safety.self_collision(arm2)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "scale" in result
    assert "reason" in result


def test_nachi_self_collision_runs():
    _, _, arm3 = make_arms()

    safety = CollisionSafety()
    result = safety.self_collision(arm3)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "scale" in result
    assert "reason" in result

# Self collision (known geometry)
def test_stretched_robot_has_no_self_collision():
    safety = CollisionSafety()

    result = safety.self_collision(straight_robot())

    assert result["safe"] is True
    assert result["scale"] == 1.0


def test_folded_robot_triggers_self_collision():
    safety = CollisionSafety(self_clearance=0.06)

    result = safety.self_collision(folded_robot(0.02))

    assert result["safe"] is False
    assert result["scale"] == 0.0
    assert result["distance"] == pytest.approx(0.02)
    assert result["link1"] == 0
    assert result["link2"] == 3


def test_self_collision_slows_down_near_limit():
    safety = CollisionSafety(self_clearance=0.06, slow_factor=2.0)

    result = safety.self_collision(folded_robot(0.09))

    assert result["safe"] is True
    assert result["scale"] == pytest.approx(0.5)


def test_zero_length_link_does_not_cause_false_self_collision():
    safety = CollisionSafety()

    result = safety.self_collision(zero_length_robot())

    assert result["safe"] is True


def test_self_collision_accepts_wrapper_class():
    safety = CollisionSafety()

    result = safety.self_collision(FakeArm(straight_robot()))

    assert result["safe"] is True

# Robot-to-robot collision
def test_kawasaki_staubli_collision_check_runs():
    arm1, arm2, _ = make_arms()
    safety = CollisionSafety()

    result = safety.robot_robot(arm1, arm2)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "scale" in result
    assert "reason" in result


def test_staubli_nachi_collision_check_runs():
    _, arm2, arm3 = make_arms()

    safety = CollisionSafety()
    result = safety.robot_robot(arm2, arm3)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "distance" in result
    assert "scale" in result
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
    assert "scale" in result
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

# Robot-to-robot collision (known geometry)
def test_far_robots_are_safe():
    safety = CollisionSafety(robot_clearance=0.10)

    result = safety.robot_robot(vertical_robot(0.0), vertical_robot(1.0))

    assert result["safe"] is True
    assert result["distance"] == pytest.approx(1.0)
    assert result["scale"] == 1.0


def test_very_close_robots_are_unsafe():
    safety = CollisionSafety(robot_clearance=0.10)

    result = safety.robot_robot(vertical_robot(0.0), vertical_robot(0.05))

    assert result["safe"] is False
    assert result["scale"] == 0.0
    assert "link1" in result
    assert "link2" in result


def test_robot_robot_slows_down_near_limit():
    safety = CollisionSafety(robot_clearance=0.10, slow_factor=2.0)

    result = safety.robot_robot(vertical_robot(0.0), vertical_robot(0.15))

    assert result["safe"] is True
    assert result["scale"] == pytest.approx(0.5)


def test_robot_robot_accepts_wrapper_classes():
    safety = CollisionSafety()

    result = safety.robot_robot(FakeArm(vertical_robot(0.0)), FakeArm(vertical_robot(1.0)))

    assert result["safe"] is True


def test_robot_robot_custom_clearance_overrides_default():
    safety = CollisionSafety(robot_clearance=0.10)

    result = safety.robot_robot(vertical_robot(0.0), vertical_robot(0.5), clearance=1.0)

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
    assert "scale" in result
    assert "reason" in result

# Obstacle collision (known geometry)
def test_far_sphere_is_safe():
    safety = CollisionSafety(obstacle_clearance=0.05)

    result = safety.sphere_obstacle(vertical_robot(0.0), [1.0, 0.0, 0.5], 0.10)

    assert result["safe"] is True
    assert result["distance"] == pytest.approx(0.9)
    assert result["scale"] == 1.0


def test_sphere_touching_robot_is_unsafe():
    safety = CollisionSafety(obstacle_clearance=0.05)

    result = safety.sphere_obstacle(vertical_robot(0.0), [0.12, 0.0, 0.5], 0.10)

    assert result["safe"] is False
    assert result["scale"] == 0.0
    assert result["link"] == 0


def test_sphere_inside_robot_link_is_unsafe():
    safety = CollisionSafety()

    result = safety.sphere_obstacle(vertical_robot(0.0), [0.0, 0.0, 0.5], 0.10)

    assert result["safe"] is False
    assert result["distance"] < 0


def test_sphere_obstacle_slows_down_near_limit():
    safety = CollisionSafety(obstacle_clearance=0.05, slow_factor=2.0)

    result = safety.sphere_obstacle(vertical_robot(0.0), [0.175, 0.0, 0.5], 0.10)

    assert result["safe"] is True
    assert result["scale"] == pytest.approx(0.5)


def test_sphere_beyond_link_end_uses_endpoint():
    safety = CollisionSafety(obstacle_clearance=0.05)

    result = safety.sphere_obstacle(vertical_robot(0.0), [0.0, 0.0, 2.0], 0.10)

    assert result["distance"] == pytest.approx(0.9)

# Merge results
def test_merge_unsafe_result_wins():
    merged = CollisionSafety.merge(
        {"safe": True, "scale": 1.0},
        {"safe": False, "scale": 0.0},
        {"safe": True, "scale": 0.3},
    )

    assert merged["safe"] is False


def test_merge_lowest_scale_wins_when_all_safe():
    merged = CollisionSafety.merge(
        {"safe": True, "scale": 1.0},
        {"safe": True, "scale": 0.3},
        {"safe": True, "scale": 0.7},
    )

    assert merged["scale"] == 0.3


def test_merge_ignores_none():
    merged = CollisionSafety.merge(
        None,
        {"safe": True, "scale": 0.4},
    )

    assert merged["scale"] == 0.4


def test_merge_result_without_scale_counts_as_full_speed():
    merged = CollisionSafety.merge(
        {"safe": True},
        {"safe": True, "scale": 0.6},
    )

    assert merged["scale"] == 0.6

# ======================================================================
# SINGULARITY SAFETY
# ======================================================================

# Creation
def test_singularity_safety_default_values():
    safety = SingularitySafety()

    assert safety.sigma_warning == 0.05
    assert safety.sigma_stop == 0.01
    assert safety.condition_warning == 100.0

#Kawwasaki RS007N

def test_kawasaki_singularity():
    arm = KawasakiPickPlace()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert isinstance(result, dict)
    assert isinstance(result["safe"], bool)
    assert result["level"] in ["SAFE","WARNING","STOP"]
    assert result["sigma_min"] >= 0
    assert 0 <= result["rank"] <= 6

#Staubli TX2-60

def test_staubli_singularity():
    arm = StaubliShapeDraw()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "level" in result
    assert "sigma_min" in result
    assert "condition_number" in result
    assert "manipulability" in result
    assert "rank" in result

def test_staubli_singularity_value_valid():
    arm = StaubliShapeDraw()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert result["sigma_min"] >= 0
    assert result["rank"] >= 0  
    assert result["rank"] <= 6

    assert result ["level"] in ["SAFE", "WARNING", "STOP"]

#Nachi MZ07
def test_nachi_singularity():
    arm = NachiShapeFill()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert isinstance(result, dict)
    assert "safe" in result
    assert "level" in result
    assert "sigma_min" in result
    assert "condition_number" in result
    assert "manipulability" in result
    assert "rank" in result

def test_nachi_singularity_value_valid():
    arm = NachiShapeFill()
    safety = SingularitySafety()

    result = safety.check(arm)

    assert result["sigma_min"] >= 0
    assert result["rank"] >= 0
    assert result["rank"] <= 6

    assert result ["level"] in ["SAFE", "WARNING", "STOP"]

#ARTIFICIAL SINGULARITY TESTS

def test_singularity_metrics_at_zero_configuration():

    """
    Check that the singularity monitor can analyse q = 0 without crashing.

    This does not assume q = 0 is necessarily singular for every robot.

    """
    arm = KawasakiPickPlace()
    safety = SingularitySafety()

    q_zero = np.zeros(6)
    result = safety.check(arm, q_zero)

    assert isinstance(result, dict)
    assert result["sigma_min"] >= 0
    assert result["rank"] >= 0
    assert result["rank"] <= 6 

#KNOWN JACOBIAN TESTS (no hardware needed)

def test_singularity_metrics_identity_jacobian():
    safety = SingularitySafety()

    result = safety.metrics(jacobian_robot(np.eye(6)))

    assert result["sigma_min"] == pytest.approx(1.0)
    assert result["sigma_max"] == pytest.approx(1.0)
    assert result["condition_number"] == pytest.approx(1.0)
    assert result["manipulability"] == pytest.approx(1.0)
    assert result["rank"] == 6


def test_singularity_metrics_rank_deficient_jacobian():
    safety = SingularitySafety()

    result = safety.metrics(jacobian_robot(np.diag([1, 1, 1, 1, 1, 0])))

    assert result["rank"] == 5
    assert result["sigma_min"] == pytest.approx(0.0, abs=1e-12)
    assert result["condition_number"] == np.inf


def test_singularity_safe_configuration():
    safety = SingularitySafety()

    result = safety.check(jacobian_robot(np.eye(6)))

    assert result["safe"] is True
    assert result["level"] == "SAFE"


def test_singularity_warning_from_small_sigma():
    safety = SingularitySafety()

    result = safety.check(jacobian_robot(np.diag([1, 1, 1, 1, 1, 0.03])))

    assert result["safe"] is True
    assert result["level"] == "WARNING"


def test_singularity_warning_from_condition_number():
    safety = SingularitySafety()

    # sigma_min is above the warning limit, but condition number is above 100
    result = safety.check(jacobian_robot(np.diag([10, 10, 10, 10, 10, 0.09])))

    assert result["safe"] is True
    assert result["level"] == "WARNING"


def test_singularity_stop_from_small_sigma():
    safety = SingularitySafety()

    result = safety.check(jacobian_robot(np.diag([1, 1, 1, 1, 1, 0.005])))

    assert result["safe"] is False
    assert result["level"] == "STOP"


def test_singularity_stop_from_rank_loss():
    safety = SingularitySafety()

    result = safety.check(jacobian_robot(np.diag([1, 1, 1, 1, 1, 0])))

    assert result["safe"] is False
    assert result["level"] == "STOP"


def test_singularity_non_6_dof_jacobian_is_not_flagged():
    safety = SingularitySafety()

    # 3x3 Jacobian with full rank must not be treated as rank loss
    result = safety.check(jacobian_robot(np.eye(3)))

    assert result["rank"] == 3
    assert result["level"] == "SAFE"


def test_singularity_custom_thresholds():
    safety = SingularitySafety(sigma_warning=0.5, sigma_stop=0.2)

    result = safety.check(jacobian_robot(np.diag([1, 1, 1, 1, 1, 0.3])))

    assert result["level"] == "WARNING"


def test_singularity_check_accepts_wrapper_class():
    safety = SingularitySafety()

    result = safety.check(FakeArm(jacobian_robot(np.eye(6))))

    assert result["level"] == "SAFE"

# ======================================================================
# ROBOT SAFETY CONTROLLER
# ======================================================================

# CREATION
def test_robot_safety_controller_created():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    assert controller is not None

# NORMAL VELOCITY
def test_normal_velocity():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert velocity == 1.0

# MANUAL VELOCITY CONTROL
def test_manual_velocity_reduction():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    controller.set_manual_scale(0.7)

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert velocity == 0.7

# MANUAL VELOCITY CLAMP
def test_manual_velocity_cannot_exceed_100_percent():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    controller.set_manual_scale(1.5)

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert velocity == 1.0


def test_manual_velocity_cannot_be_negative():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    controller.set_manual_scale(-0.5)

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert velocity == 0.0

# SINGULARITY WARNING
def test_warning_singularity_reduces_velocity():
    controller = RobotSafetyController(
        max_velocity=1.0,
        warning_scale=0.5
    )

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        warning_singularity()
    )

    assert velocity == 0.5

# MANUAL SETTING CANNOT OVERRIDE SAFETY
def test_manual_speed_cannot_override_singularity_limit():
    controller = RobotSafetyController(
        max_velocity=1.0,
        warning_scale=0.5
    )

    controller.set_manual_scale(0.9)

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        warning_singularity()
    )

    assert velocity == 0.5

# LOWER MANUAL SPEED IS ALLOWED
def test_lower_manual_speed_is_respected():
    controller = RobotSafetyController(
        max_velocity=1.0,
        warning_scale=0.5
    )

    controller.set_manual_scale(0.3)

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        warning_singularity()
    )

    assert velocity == 0.3

# OFFSET LINK ROBOT
def offset_link_robot():
    # Short 0.02 m offset link between two vertical links (like the Staubli elbow)
    return FakeRobot(points=[[0, 0, 0], [0, 0, 1], [0.02, 0, 1], [0.02, 0, 2]])


def test_short_offset_link_does_not_cause_false_self_collision():
    safety = CollisionSafety(self_clearance=0.06)

    result = safety.self_collision(offset_link_robot())

    assert result["safe"] is True

# COLLISION STOP
def test_collision_sets_velocity_to_zero():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    velocity = controller.get_allowed_velocity(
        unsafe_collision(),
        safe_singularity()
    )

    assert velocity == 0.0

# COLLISION SLOW DOWN (near but not colliding)
def test_close_collision_reduces_velocity():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    velocity = controller.get_allowed_velocity(
        slow_collision(0.4),
        safe_singularity()
    )

    assert velocity == pytest.approx(0.4)


def test_manual_speed_cannot_override_collision_slow_down():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    controller.set_manual_scale(0.9)

    velocity = controller.get_allowed_velocity(
        slow_collision(0.4),
        safe_singularity()
    )

    assert velocity == pytest.approx(0.4)


def test_lower_manual_speed_is_respected_near_collision():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    controller.set_manual_scale(0.3)

    velocity = controller.get_allowed_velocity(
        slow_collision(0.4),
        safe_singularity()
    )

    assert velocity == pytest.approx(0.3)


def test_lowest_safety_scale_is_used():
    controller = RobotSafetyController(
        max_velocity=1.0,
        warning_scale=0.5
    )

    # Singularity warning (0.5) is lower than collision scale (0.8)
    velocity = controller.get_allowed_velocity(
        slow_collision(0.8),
        warning_singularity()
    )

    assert velocity == pytest.approx(0.5)

    # Collision scale (0.2) is lower than singularity warning (0.5)
    velocity = controller.get_allowed_velocity(
        slow_collision(0.2),
        warning_singularity()
    )

    assert velocity == pytest.approx(0.2)


def test_collision_without_scale_is_full_speed():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    velocity = controller.get_allowed_velocity(
        {"safe": True},
        safe_singularity()
    )

    assert velocity == 1.0

# SEVERE SINGULARITY STOP
def test_severe_singularity_sets_velocity_to_zero():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        stop_singularity()
    )

    assert velocity == 0.0

# E-STOP
def test_estop_sets_velocity_to_zero():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    controller.trigger_estop()

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert velocity == 0.0


def test_estop_state_is_active():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    controller.trigger_estop()

    assert controller.is_estopped() is True


def test_estop_state_is_inactive_by_default():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    assert controller.is_estopped() is False

# E-STOP RESET
def test_estop_reset_restores_motion():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    controller.trigger_estop()
    controller.reset_estop()

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert controller.is_estopped() is False
    assert velocity == 1.0

# PRIORITY
def test_estop_has_priority_over_everything():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    controller.set_manual_scale(1.0)
    controller.trigger_estop()

    velocity = controller.get_allowed_velocity(
        slow_collision(1.0),
        safe_singularity()
    )

    assert velocity == 0.0


def test_collision_and_singularity_stop_together():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    velocity = controller.get_allowed_velocity(
        unsafe_collision(),
        stop_singularity()
    )

    assert velocity == 0.0


def test_collision_stop_overrides_singularity_warning():
    controller = RobotSafetyController(
        max_velocity=1.0
    )

    velocity = controller.get_allowed_velocity(
        unsafe_collision(),
        warning_singularity()
    )

    assert velocity == 0.0

# MAX VELOCITY
def test_allowed_velocity_scales_with_max_velocity():
    controller = RobotSafetyController(
        max_velocity=2.0
    )

    controller.set_manual_scale(0.5)

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert velocity == pytest.approx(1.0)


def test_velocity_scale_matches_allowed_velocity():
    controller = RobotSafetyController(
        max_velocity=2.0
    )

    scale = controller.get_velocity_scale(
        slow_collision(0.25),
        safe_singularity()
    )

    velocity = controller.get_allowed_velocity(
        slow_collision(0.25),
        safe_singularity()
    )

    assert scale == pytest.approx(0.25)
    assert velocity == pytest.approx(0.5)

# INDIVIDUAL ROBOT LIMITS
def test_kawasaki_velocity_limit():
    controller = RobotSafetyController(
        max_velocity=ROBOT_VELOCITY_LIMITS["Kawasaki"]
    )

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert velocity <= ROBOT_VELOCITY_LIMITS["Kawasaki"]


def test_staubli_velocity_limit():
    controller = RobotSafetyController(
        max_velocity=ROBOT_VELOCITY_LIMITS["Staubli"]
    )

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert velocity <= ROBOT_VELOCITY_LIMITS["Staubli"]


def test_nachi_velocity_limit():
    controller = RobotSafetyController(
        max_velocity=ROBOT_VELOCITY_LIMITS["Nachi"]
    )

    velocity = controller.get_allowed_velocity(
        safe_collision(),
        safe_singularity()
    )

    assert velocity <= ROBOT_VELOCITY_LIMITS["Nachi"]

# ROBOT NAME USES DEFAULT LIMIT
def test_robot_name_uses_default_velocity_limit():
    for name in ["Kawasaki", "Staubli", "Nachi"]:
        controller = RobotSafetyController(
            max_velocity=5.0,
            robot_name=name
        )

        assert controller.max_velocity == ROBOT_VELOCITY_LIMITS[name]


def test_unknown_robot_name_keeps_given_velocity():
    controller = RobotSafetyController(
        max_velocity=5.0,
        robot_name="Unknown"
    )

    assert controller.max_velocity == 5.0

# ======================================================================
# RMRC WITH SAFETY SCALING
# ======================================================================

def test_rmrc_normal_velocity_uses_pseudo_inverse():
    controller = RobotSafetyController(max_velocity=1.0)
    robot = jacobian_robot(np.eye(6))
    v = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    qdot = controller.rmrc_joint_velocity(robot, v, safe_collision(), safe_singularity())

    assert np.allclose(qdot, v)


def test_rmrc_applies_manual_scale():
    controller = RobotSafetyController(max_velocity=1.0)
    controller.set_manual_scale(0.5)
    robot = jacobian_robot(np.eye(6))
    v = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    qdot = controller.rmrc_joint_velocity(robot, v, safe_collision(), safe_singularity())

    assert np.allclose(qdot, 0.5 * v)


def test_rmrc_applies_collision_scale():
    controller = RobotSafetyController(max_velocity=1.0)
    robot = jacobian_robot(np.eye(6))
    v = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    qdot = controller.rmrc_joint_velocity(robot, v, slow_collision(0.4), safe_singularity())

    assert np.allclose(qdot, 0.4 * v)


def test_rmrc_limits_linear_speed_to_max_velocity():
    controller = RobotSafetyController(max_velocity=1.0)
    robot = jacobian_robot(np.eye(6))
    v = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    qdot = controller.rmrc_joint_velocity(robot, v, safe_collision(), safe_singularity())

    assert qdot[0] == pytest.approx(1.0)


def test_rmrc_stops_on_estop():
    controller = RobotSafetyController(max_velocity=1.0)
    controller.trigger_estop()
    robot = jacobian_robot(np.eye(6))
    v = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    qdot = controller.rmrc_joint_velocity(robot, v, safe_collision(), safe_singularity())

    assert np.allclose(qdot, np.zeros(6))


def test_rmrc_stops_on_collision():
    controller = RobotSafetyController(max_velocity=1.0)
    robot = jacobian_robot(np.eye(6))
    v = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    qdot = controller.rmrc_joint_velocity(robot, v, unsafe_collision(), safe_singularity())

    assert np.allclose(qdot, np.zeros(6))


def test_rmrc_stops_on_severe_singularity():
    controller = RobotSafetyController(max_velocity=1.0)
    robot = jacobian_robot(np.eye(6))
    v = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    qdot = controller.rmrc_joint_velocity(robot, v, safe_collision(), stop_singularity())

    assert np.allclose(qdot, np.zeros(6))


def test_rmrc_rotation_only_request_is_stopped_by_estop():
    controller = RobotSafetyController(max_velocity=1.0)
    controller.trigger_estop()
    robot = jacobian_robot(np.eye(6))
    v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    qdot = controller.rmrc_joint_velocity(robot, v, safe_collision(), safe_singularity())

    assert np.allclose(qdot, np.zeros(6))


def test_rmrc_warning_uses_damped_least_squares():
    controller = RobotSafetyController(max_velocity=1.0, warning_scale=0.5)
    robot = jacobian_robot(np.eye(6))
    v = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    damping = 0.05

    qdot = controller.rmrc_joint_velocity(
        robot, v, safe_collision(), warning_singularity(), damping=damping
    )

    # J = I  ->  qdot = scale * v / (1 + damping^2)
    expected = 0.5 * v / (1.0 + damping ** 2)

    assert np.allclose(qdot, expected)


def test_rmrc_damped_solution_stays_finite_near_singularity():
    controller = RobotSafetyController(max_velocity=1.0, warning_scale=0.5)
    robot = jacobian_robot(np.diag([1, 1, 1, 1, 1, 1e-9]))
    v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1])

    qdot = controller.rmrc_joint_velocity(robot, v, safe_collision(), warning_singularity())

    assert np.all(np.isfinite(qdot))
    assert np.max(np.abs(qdot)) < 1.0


def test_rmrc_returns_one_value_per_joint():
    controller = RobotSafetyController(max_velocity=1.0)
    robot = jacobian_robot(np.eye(6))
    v = np.zeros(6)

    qdot = controller.rmrc_joint_velocity(robot, v, safe_collision(), safe_singularity())

    assert qdot.shape == (6,)


def test_rmrc_accepts_wrapper_class():
    controller = RobotSafetyController(max_velocity=1.0)
    arm = FakeArm(jacobian_robot(np.eye(6)))
    v = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    qdot = controller.rmrc_joint_velocity(arm, v, safe_collision(), safe_singularity())

    assert np.allclose(qdot, v)


def test_rmrc_accepts_explicit_joint_configuration():
    controller = RobotSafetyController(max_velocity=1.0)
    robot = jacobian_robot(np.eye(6))
    v = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    qdot = controller.rmrc_joint_velocity(
        robot, v, safe_collision(), safe_singularity(), q=np.zeros(6)
    )

    assert np.allclose(qdot, v)