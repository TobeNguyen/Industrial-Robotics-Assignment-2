import os
import sys

# Allow tests/ to import files from AT 2/
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from safety import (
    RobotSafetyController,
    ROBOT_VELOCITY_LIMITS
)

# Helper safety results
def safe_collision():
    return {
        "safe": True
    }


def unsafe_collision():
    return {
        "safe": False
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