# Including libraries
import math
import os
import sys
import tempfile
import time
import types
import unittest
import urllib.error
import urllib.parse
import urllib.request
from unittest.mock import patch

import numpy as np

import gui
from safety import RobotSafetyController

SAFE_COLLISION = {'safe': True, 'distance': 1.0, 'scale': 1.0, 'reason': 'No collision risk'}
UNSAFE_COLLISION = {'safe': False, 'distance': 0.02, 'scale': 0.0, 'reason': 'Self-collision risk'}
SAFE_SINGULARITY = {'safe': True, 'level': 'SAFE', 'sigma_min': 1.0}


class FakeEvent:
    def __init__(self, number=0):
        self.number = number

class FakeWidget:
    def __init__(self, *args, **kwargs):
        self.desc = kwargs.get('desc', '')
        self.args = args
        self.kwargs = kwargs

class FakeEnv:
    def __init__(self):
        self.added = []

    def add(self, widget):
        self.added.append(widget)
        return widget

# Minimal stand-in for a robot: 6 joints, joint limits of -170.5 to 100.2 degrees
class FakeRobot:
    def __init__(self, n=6, lim_deg=(-170.5, 100.2)):
        self.n = n
        self.q = np.zeros(n)
        self.qlim = np.array([[math.radians(lim_deg[0])] * n,
                              [math.radians(lim_deg[1])] * n])

# Minimal stand-in for the arm classes: only what the joint sliders use (robot.q, robot.qlim, robot.n, HOME)
class FakeJointArm:
    def __init__(self, home_deg=(0, 20, 70, 0, 90, 0)):
        self.robot = FakeRobot()
        self.HOME = np.deg2rad(home_deg)
        self.robot.q = self.HOME.copy()

def make_controllers(n=3):
    return [RobotSafetyController(robot_name=name) for name in gui.ARM_NAMES[:n]]

def reset_gui_state():
    gui.stop_upload_server()
    gui._selected_mode = 'random'
    gui._selected_shape = None
    gui._selected_colour = None
    gui._rgb[:] = gui.DEFAULT_RGB
    gui._pending_command = None
    gui._uploaded_files.queue.clear()
    gui._arms.clear()
    gui._joint_sliders.clear()
    gui._controllers.clear()
    gui._widgets.clear()
    gui._last_safety.clear()
    gui._selected_arm = 0
    gui._cycle_running = False
    gui._abort_requested = False
    gui._reset_requested = False
    gui._crate = None                   # ADDED
    gui._crate_active = False           # ADDED
    gui._scene = {}                     # ADDED
    gui._curtain_broken = False         # ADDED
    gui._rmrc_v[:] = [0.0] * 6
    gui._rmrc_axis = 0
    gui._rmrc_slider = None
    gui._speed_slider = None
    gui._last_label_time = 0.0
    gui._last_jog_time = time.time()
    gui._last_jog_sync = 0.0
    gui.image_models.clear()
    gui.image_strokes.clear()
    gui.status.update(process='READY', mode=None, shape=None, colour=None,
                    image_name=None, image_ready=False, servo=None, log=[])

# Base class: starts every test from a clean gui state, with gui.evaluate_safety replaced by
# a "everything is safe" stub (the fake arms have no kinematics). Change self.safety_mock to test other results.
# The collision / singularity / speed-scale logic itself is tested in test_safety.py, not here.
class SafetyPatchedCase(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        patcher = patch('gui.evaluate_safety', return_value=(SAFE_COLLISION, SAFE_SINGULARITY))
        self.safety_mock = patcher.start()
        self.addCleanup(patcher.stop)


class TestSelection(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_select_mode_updates_state(self):
        gui.select_mode('basic')
        self.assertEqual(gui._selected_mode, 'basic')

    def test_select_shape_updates_state(self):
        gui.select_shape('square')
        self.assertEqual(gui._selected_shape, 'square')

    def test_select_colour_updates_state(self):
        gui.select_colour('red')
        self.assertEqual(gui._selected_colour, 'red')

    def test_selections_are_logged(self):
        gui.select_mode('image')
        self.assertIn('Mode set to image', gui.status['log'][-1])

class TestSelectRgb(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_select_rgb_updates_one_channel(self):
        gui.select_rgb(0, 255)
        self.assertEqual(gui._selected_colour, (255, 40, 40))

    def test_select_rgb_keeps_other_channels(self):
        gui.select_rgb(0, 10)
        gui.select_rgb(2, 200)
        self.assertEqual(gui._selected_colour, (10, 40, 200))

    def test_select_rgb_converts_value_to_int(self):
        gui.select_rgb(1, 99.7)
        self.assertEqual(gui._selected_colour, (200, 99, 40))

class TestJointControl(SafetyPatchedCase):
    def setUp(self):
        super().setUp()
        self.arm = FakeJointArm()

    def test_select_joint_sets_the_joint_in_radians(self):
        gui.select_joint(self.arm, 2, 45)
        self.assertAlmostEqual(self.arm.robot.q[2], math.radians(45), places=9)

    def test_select_joint_leaves_other_joints_alone(self):
        gui.select_joint(self.arm, 2, 45)
        expected = self.arm.HOME.copy()
        expected[2] = math.radians(45)
        np.testing.assert_allclose(self.arm.robot.q, expected, atol=1e-12)

    def test_select_joint_accepts_fractional_degrees(self):
        gui.select_joint(self.arm, 0, 10.5)
        self.assertAlmostEqual(self.arm.robot.q[0], math.radians(10.5), places=9)

    def test_select_joint_clamps_to_the_joint_limits(self):
        gui.select_joint(self.arm, 0, 500)
        self.assertAlmostEqual(self.arm.robot.q[0], math.radians(100.2), places=9)
        gui.select_joint(self.arm, 0, -500)
        self.assertAlmostEqual(self.arm.robot.q[0], math.radians(-170.5), places=9)

    def test_select_joint_ignored_while_running(self):
        gui._cycle_running = True
        gui.select_joint(self.arm, 2, 45)
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)

    def test_select_joint_works_again_after_the_cycle_completes(self):
        gui._cycle_running = True
        gui.select_joint(self.arm, 0, 30)
        gui.update_status(process='COMPLETE')
        gui.select_joint(self.arm, 0, 30)
        self.assertAlmostEqual(self.arm.robot.q[0], math.radians(30), places=9)

    def test_sync_joint_sliders_follow_the_arm(self):
        sliders = [FakeWidget() for _ in range(6)]
        gui._arms[:] = [self.arm]
        gui._joint_sliders[:] = sliders
        self.arm.robot.q = np.deg2rad([0, 33.4, 70, 0, 90, 0])
        gui.sync_joint_sliders()
        self.assertEqual(sliders[1].value, 33)

    def test_sync_joint_sliders_follow_the_arm_limits(self):
        sliders = [FakeWidget() for _ in range(6)]
        gui._arms[:] = [self.arm]
        gui._joint_sliders[:] = sliders
        gui.sync_joint_sliders()
        self.assertEqual((sliders[0].min, sliders[0].max), (-171, 101))

    def test_sync_joint_sliders_without_arms_does_not_raise(self):
        gui.sync_joint_sliders()

    def test_select_arm_points_the_sliders_at_that_arm(self):
        arm2 = FakeJointArm(home_deg=(0, 30, 60, 0, 45, 0))
        arm2.robot.qlim = np.array([[math.radians(-90)] * 6, [math.radians(90)] * 6])
        sliders = [FakeWidget() for _ in range(6)]
        gui._arms[:] = [self.arm, arm2]
        gui._joint_sliders[:] = sliders
        gui.select_arm(1)
        self.assertEqual(gui._selected_arm, 1)
        self.assertEqual((sliders[1].min, sliders[1].max, sliders[1].value), (-90, 90, 30))

    def test_select_arm_is_logged(self):
        gui._arms[:] = [self.arm, FakeJointArm()]
        gui._joint_sliders[:] = [FakeWidget() for _ in range(6)]
        gui.select_arm(1)
        self.assertIn('Staubli', gui.status['log'][-1])

    def test_select_arm_ignores_an_unknown_index(self):
        gui._arms[:] = [self.arm]
        gui._joint_sliders[:] = [FakeWidget() for _ in range(6)]
        gui.select_arm(5)
        self.assertEqual(gui._selected_arm, 0)

    def test_select_arm_updates_the_adjust_label(self):
        gui._arms[:] = [self.arm, FakeJointArm()]
        gui._joint_sliders[:] = [FakeWidget() for _ in range(6)]
        gui._widgets['adjust_label'] = FakeWidget(desc='Adjusting: Kawasaki')
        gui.select_arm(1)
        self.assertEqual(gui._widgets['adjust_label'].desc, 'Adjusting: Staubli')

class TestResetArms(SafetyPatchedCase):
    def setUp(self):
        super().setUp()
        self.arm = FakeJointArm()
        gui._arms[:] = [self.arm]
        gui._controllers[:] = make_controllers(1)

    def test_reset_arms_returns_every_arm_to_home(self):
        arm2 = FakeJointArm(home_deg=(0, 30, 70, -180, -80, 0))
        for arm in (self.arm, arm2):
            arm.robot.q = np.zeros(6)
        gui._arms[:] = [self.arm, arm2]
        ok = gui.reset_arms()
        self.assertTrue(ok)
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)
        np.testing.assert_allclose(arm2.robot.q, arm2.HOME, atol=1e-12)

    def test_reset_arms_does_not_alias_the_home_vector(self):
        gui.reset_arms()
        self.arm.robot.q[0] = 1.0
        self.assertEqual(self.arm.HOME[0], 0.0)

    def test_reset_arms_resets_the_sliders(self):
        sliders = [FakeWidget() for _ in range(6)]
        gui._joint_sliders[:] = sliders
        self.arm.robot.q = np.zeros(6)
        gui.reset_arms()
        self.assertEqual(sliders[1].value, 20)

    def test_reset_arms_is_logged(self):
        gui.reset_arms()
        self.assertIn('reset to home', gui.status['log'][-1])

    def test_reset_arms_after_hand_moved_joints_goes_home(self):
        gui.select_joint(self.arm, 1, 60)
        gui.reset_arms()
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)

    def test_reset_arms_works_after_a_completed_process(self):
        gui.update_status(process='COMPLETE')
        self.arm.robot.q = np.zeros(6)
        self.assertTrue(gui.reset_arms())
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)

    def test_reset_arms_blocked_while_running(self):
        self.arm.robot.q = np.zeros(6)
        gui._cycle_running = True
        ok = gui.reset_arms()
        self.assertFalse(ok)
        np.testing.assert_allclose(self.arm.robot.q, np.zeros(6), atol=1e-12)
        self.assertIn('Cannot reset', gui.status['log'][-1])

    def test_reset_arms_refused_during_estop(self):
        gui.request_stop()
        self.arm.robot.q = np.zeros(6)
        self.assertFalse(gui.reset_arms())
        np.testing.assert_allclose(self.arm.robot.q, np.zeros(6), atol=1e-12)
        self.assertTrue(gui._estopped())
        self.assertIn('RESET PROCESS', gui.status['log'][-1])

    def test_reset_arms_does_not_request_a_scene_reset(self):
        gui.reset_arms()
        self.assertFalse(gui.consume_reset_request())

    def test_reset_arms_does_not_abort_anything(self):
        gui.reset_arms()
        self.assertFalse(gui.abort_requested())

    def test_reset_arms_without_arms_does_not_raise(self):
        gui._arms.clear()
        self.assertTrue(gui.reset_arms())

class TestRequestRun(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_random_mode_queues_command(self):
        gui.select_mode('random')
        gui.select_colour('blue')
        ok = gui.request_run()
        self.assertTrue(ok)
        command = gui.get_pending_command()
        self.assertEqual(command['mode'], 'random')
        self.assertEqual(command['colour'], 'blue')

    def test_rgb_colour_is_passed_in_command(self):
        gui.select_mode('random')
        gui.select_rgb(0, 12)
        gui.select_rgb(1, 34)
        gui.select_rgb(2, 56)
        gui.request_run()
        command = gui.get_pending_command()
        self.assertEqual(command['colour'], (12, 34, 56))

    def test_basic_mode_without_shape_fails(self):
        gui.select_mode('basic')
        ok = gui.request_run()
        self.assertFalse(ok)
        self.assertIsNone(gui.get_pending_command())

    def test_basic_mode_with_shape_queues_command(self):
        gui.select_mode('basic')
        gui.select_shape('triangle')
        ok = gui.request_run()
        self.assertTrue(ok)
        command = gui.get_pending_command()
        self.assertEqual(command['shape'], 'triangle')

    def test_image_mode_without_ready_image_fails(self):
        gui.select_mode('image')
        gui.update_status(image_ready=False)
        ok = gui.request_run()
        self.assertFalse(ok)

    def test_image_mode_with_ready_image_queues_command(self):
        gui.select_mode('image')
        gui.update_status(image_ready=True)
        ok = gui.request_run()
        self.assertTrue(ok)

    def test_request_run_updates_status_to_running(self):
        gui.select_mode('random')
        gui.request_run()
        snap = gui.get_status_snapshot()
        self.assertEqual(snap['process'], 'RUNNING')

    def test_request_run_marks_the_cycle_as_running(self):
        gui.select_mode('random')
        gui.request_run()
        self.assertTrue(gui._cycle_running)

    def test_request_run_refused_during_estop(self):
        gui._controllers[:] = make_controllers()
        gui.request_stop()
        gui.select_mode('random')
        self.assertFalse(gui.request_run())
        self.assertIsNone(gui.get_pending_command())
        self.assertIn('START blocked', gui.status['log'][-1])      # CHANGED

class TestRequestStop(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        gui._controllers[:] = make_controllers()

    def test_request_stop_triggers_estop_on_every_controller(self):
        gui.request_stop()
        self.assertTrue(all(c.is_estopped() for c in gui._controllers))

    def test_request_stop_does_not_queue_a_command(self):
        gui.request_stop()
        self.assertIsNone(gui.get_pending_command())

    def test_request_stop_clears_a_queued_run(self):
        gui.select_mode('random')
        gui.request_run()
        gui.request_stop()
        self.assertIsNone(gui.get_pending_command())

    def test_request_stop_updates_status(self):
        gui.request_stop()
        self.assertEqual(gui.get_status_snapshot()['process'], 'E-STOP')

    def test_request_stop_is_logged(self):
        gui.request_stop()
        self.assertIn('E-STOP active', gui.status['log'][-1])      # CHANGED

    def test_request_stop_without_controllers_does_not_raise(self):
        gui._controllers.clear()
        gui.request_stop()
        self.assertEqual(gui.get_status_snapshot()['process'], 'E-STOP')

class TestGetPendingCommand(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_get_pending_command_clears_after_read(self):
        gui.select_mode('random')
        gui.request_run()
        first = gui.get_pending_command()
        second = gui.get_pending_command()
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_no_pending_command_returns_none(self):
        self.assertIsNone(gui.get_pending_command())

# Only the wiring of gui.safety_scale is tested here; the scale rules are tested in test_safety.py
class TestSafetyScale(SafetyPatchedCase):
    def setUp(self):
        super().setUp()
        self.arm = FakeJointArm()
        self.controller = RobotSafetyController()

    def test_result_of_evaluate_safety_reaches_the_controller(self):
        self.safety_mock.return_value = (UNSAFE_COLLISION, SAFE_SINGULARITY)
        self.assertEqual(gui.safety_scale(self.arm, None, self.controller), 0.0)

    def test_result_is_recorded_for_a_registered_arm(self):
        gui._arms[:] = [self.arm]
        gui.safety_scale(self.arm, None, self.controller)
        collision, singularity, set_scale, allowed = gui._last_safety[0]
        self.assertEqual(set_scale, 1.0)
        self.assertEqual(allowed, 1.0)

    def test_selected_arm_controller_is_used_when_none_is_given(self):
        gui._arms[:] = [self.arm]
        gui._controllers[:] = [self.controller]
        self.controller.trigger_estop()
        self.assertEqual(gui.safety_scale(self.arm), 0.0)

    def _arm_in_cycle(self):
        gui._arms[:] = [self.arm]
        gui._controllers[:] = [self.controller]
        gui._cycle_running = True

    def test_collision_during_a_cycle_latches_estop(self):
        self._arm_in_cycle()
        self.safety_mock.return_value = (UNSAFE_COLLISION, SAFE_SINGULARITY)
        self.assertEqual(gui.safety_scale(self.arm, None, self.controller), 0.0)
        self.assertTrue(gui._estopped())
        self.assertEqual(gui.get_status_snapshot()['process'], 'E-STOP')
        self.assertTrue(any('Safety stop' in m for m in gui.status['log']))

    def test_severe_singularity_during_a_cycle_latches_estop(self):
        self._arm_in_cycle()
        stop = {'safe': False, 'level': 'STOP', 'reason': 'Severe singularity detected', 'sigma_min': 0.0}
        self.safety_mock.return_value = (SAFE_COLLISION, stop)
        gui.safety_scale(self.arm, None, self.controller)
        self.assertTrue(gui._estopped())

    def test_collision_outside_a_cycle_does_not_latch_estop(self):
        self._arm_in_cycle()
        gui._cycle_running = False
        self.safety_mock.return_value = (UNSAFE_COLLISION, SAFE_SINGULARITY)
        self.assertEqual(gui.safety_scale(self.arm, None, self.controller), 0.0)
        self.assertFalse(gui._estopped())

    def test_warning_only_slows_down_and_does_not_latch_estop(self):
        self._arm_in_cycle()
        warning = {'safe': True, 'level': 'WARNING', 'sigma_min': 0.03}
        self.safety_mock.return_value = (SAFE_COLLISION, warning)
        self.assertAlmostEqual(gui.safety_scale(self.arm, None, self.controller), 0.5)
        self.assertFalse(gui._estopped())

    def test_reset_process_clears_a_safety_stop_and_aborts_the_cycle(self):
        self._arm_in_cycle()
        self.safety_mock.return_value = (UNSAFE_COLLISION, SAFE_SINGULARITY)
        gui.safety_scale(self.arm, None, self.controller)
        self.assertTrue(gui.reset_process())
        self.assertFalse(gui._estopped())
        self.assertTrue(gui.abort_requested())

class TestEvaluateSafety(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        self.a1 = FakeJointArm()
        self.a2 = FakeJointArm()
        gui._arms[:] = [self.a1, self.a2]

    def test_safe_result_reports_the_smallest_distance(self):
        with patch.object(gui._collision, 'self_collision',
                        return_value=dict(SAFE_COLLISION, distance=0.5)), \
            patch.object(gui._collision, 'robot_robot',
                        return_value=dict(SAFE_COLLISION, distance=0.2)), \
            patch.object(gui._singularity, 'check', return_value=SAFE_SINGULARITY):
            collision, _ = gui.evaluate_safety(self.a1)
        self.assertTrue(collision['safe'])
        self.assertAlmostEqual(collision['distance'], 0.2)

    def test_only_the_other_arms_are_checked_against_the_arm(self):
        with patch.object(gui._collision, 'self_collision', return_value=SAFE_COLLISION), \
            patch.object(gui._collision, 'robot_robot', return_value=SAFE_COLLISION) as robot_robot, \
            patch.object(gui._singularity, 'check', return_value=SAFE_SINGULARITY):
            gui.evaluate_safety(self.a1)
        robot_robot.assert_called_once()
        self.assertIs(robot_robot.call_args.args[0], self.a1)
        self.assertIs(robot_robot.call_args.args[1], self.a2)

class TestEstopBlocksJoints(SafetyPatchedCase):
    def setUp(self):
        super().setUp()
        self.arm = FakeJointArm()
        gui._arms[:] = [self.arm]
        gui._controllers[:] = make_controllers(1)

    def test_select_joint_blocked_during_estop(self):
        gui.request_stop()
        gui.select_joint(self.arm, 2, 45)
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)
        self.assertIn('E-STOP active', gui.status['log'][-1])

    def test_select_joint_works_after_reset_process(self):
        gui.request_stop()
        gui.reset_process()
        gui.finish_reset_process()                                  # ADDED
        gui.select_joint(self.arm, 2, 45)
        self.assertAlmostEqual(self.arm.robot.q[2], math.radians(45), places=9)

class TestResetProcess(SafetyPatchedCase):
    def setUp(self):
        super().setUp()
        self.arm = FakeJointArm()
        gui._arms[:] = [self.arm]
        gui._controllers[:] = make_controllers(1)

    def test_reset_clears_estop_and_goes_home(self):
        gui.request_stop()
        self.arm.robot.q = np.zeros(6)
        self.assertTrue(gui.reset_process())
        self.assertFalse(gui._estopped())
        gui.finish_reset_process()                                  # ADDED
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)
        self.assertEqual(gui.get_status_snapshot()['process'], 'READY')

    def test_reset_after_hand_moved_joints_goes_home(self):
        gui.select_joint(self.arm, 1, 60)
        gui.reset_process()
        gui.finish_reset_process()                                  # ADDED
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)

    def test_reset_is_refused_while_running_without_estop(self):
        gui._cycle_running = True
        self.assertFalse(gui.reset_process())
        self.assertFalse(gui.abort_requested())
        self.assertIn('Cannot reset', gui.status['log'][-1])

    def test_refused_reset_does_not_request_a_scene_reset(self):
        gui._cycle_running = True
        gui.reset_process()
        self.assertFalse(gui.consume_reset_request())

    def test_reset_after_estop_during_a_cycle_requests_abort(self):
        gui._cycle_running = True
        gui.request_stop()
        self.assertTrue(gui.reset_process())
        self.assertTrue(gui.abort_requested())
        self.assertFalse(gui._estopped())

    # CHANGED: the request now stays until finish_reset_process() is called by the main loop
    def test_reset_request_stays_until_finish_reset_process(self):
        self.assertTrue(gui.reset_process())
        self.assertTrue(gui.consume_reset_request())
        self.assertTrue(gui.consume_reset_request())
        gui.finish_reset_process()
        self.assertFalse(gui.consume_reset_request())

    def test_reset_works_after_a_completed_process(self):
        gui.update_status(process='COMPLETE')
        self.arm.robot.q = np.zeros(6)
        self.assertTrue(gui.reset_process())
        self.assertTrue(gui.consume_reset_request())                # CHANGED (moved before finish)
        gui.finish_reset_process()                                  # ADDED
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)
        self.assertEqual(gui.get_status_snapshot()['process'], 'READY')

    def test_reset_clears_the_servo_status(self):
        gui.update_status(servo='corrected 4.0 mm')
        gui.reset_process()
        self.assertIsNone(gui.get_status_snapshot()['servo'])

    def test_reset_is_logged(self):
        gui.reset_process()
        gui.finish_reset_process()                                  # ADDED
        self.assertIn('Reset complete', gui.status['log'][-1])      # CHANGED

    def test_reset_without_arms_and_controllers_does_not_raise(self):
        gui._arms.clear()
        gui._controllers.clear()
        self.assertTrue(gui.reset_process())

    def test_acknowledge_abort_returns_to_ready(self):
        gui._cycle_running = True
        gui.request_stop()
        gui.reset_process()
        gui.acknowledge_abort()
        self.assertFalse(gui.abort_requested())
        self.assertFalse(gui._cycle_running)
        self.assertEqual(gui.get_status_snapshot()['process'], 'READY')

    def test_a_new_run_is_possible_after_abort(self):
        gui._cycle_running = True
        gui.request_stop()
        gui.reset_process()
        gui.acknowledge_abort()
        gui.finish_reset_process()                                  # ADDED
        gui.select_mode('random')
        self.assertTrue(gui.request_run())

class TestSpeed(SafetyPatchedCase):
    def setUp(self):
        super().setUp()
        self.arms = [FakeJointArm(), FakeJointArm(), FakeJointArm()]
        gui._arms[:] = self.arms
        gui._joint_sliders[:] = [FakeWidget() for _ in range(6)]
        gui._controllers[:] = make_controllers()

    def test_select_speed_sets_the_selected_arm_only(self):
        gui.select_speed(50)
        self.assertAlmostEqual(gui._controllers[0].manual_scale, 0.5)
        self.assertAlmostEqual(gui._controllers[1].manual_scale, 1.0)
        self.assertAlmostEqual(gui._controllers[2].manual_scale, 1.0)

    def test_select_speed_follows_the_selected_arm(self):
        gui.select_arm(2)
        gui.select_speed(30)
        self.assertAlmostEqual(gui._controllers[2].manual_scale, 0.3)
        self.assertAlmostEqual(gui._controllers[0].manual_scale, 1.0)

    def test_speed_slider_is_synced_when_the_arm_changes(self):
        gui._speed_slider = FakeWidget()
        gui._controllers[1].set_manual_scale(0.3)
        gui.select_arm(1)
        self.assertEqual(gui._speed_slider.value, 30)

    def test_speed_slider_never_goes_below_its_minimum(self):
        gui._speed_slider = FakeWidget()
        gui._controllers[0].set_manual_scale(0.0)
        gui._sync_speed_slider()
        self.assertEqual(gui._speed_slider.value, 5)

    def test_safety_text_shows_reduced_speed(self):
        gui._last_safety[0] = (SAFE_COLLISION, SAFE_SINGULARITY, 1.0, 0.4)
        text = gui._safety_text()
        self.assertIn('set 100% -> allowed 40%', text)
        self.assertIn('(reduced)', text)

    def test_safety_text_not_reduced_when_speed_is_allowed(self):
        gui._last_safety[0] = (SAFE_COLLISION, SAFE_SINGULARITY, 0.5, 0.5)
        self.assertNotIn('(reduced)', gui._safety_text())

    def test_safety_text_shows_estop(self):
        gui.request_stop()
        self.assertIn('E-STOP ACTIVE', gui._safety_text())

    def test_safety_text_without_arms(self):
        gui._arms.clear()
        self.assertEqual(gui._safety_text(), 'Safety: -')

    def test_adjust_text_names_the_selected_robot(self):
        self.assertEqual(gui._adjust_text(), 'Adjusting: Kawasaki')
        gui.select_arm(2)
        self.assertEqual(gui._adjust_text(), 'Adjusting: Nachi')

    def test_adjust_text_without_arms(self):
        gui._arms.clear()
        self.assertEqual(gui._adjust_text(), 'Adjusting: -')

class TestRmrc(SafetyPatchedCase):
    def setUp(self):
        super().setUp()
        self.arm = FakeJointArm()
        self.controller = RobotSafetyController()
        gui._arms[:] = [self.arm]
        gui._controllers[:] = [self.controller]

    def test_select_axis_resets_slider_and_velocity(self):
        gui._rmrc_slider = FakeWidget()
        gui._rmrc_slider.value = 50
        gui.select_rmrc(50)
        gui.select_rmrc_axis(3)
        self.assertEqual(gui._rmrc_axis, 3)
        self.assertEqual(gui._rmrc_slider.value, 0)
        self.assertEqual(gui._rmrc_v, [0.0] * 6)

    def test_linear_axis_is_in_mm_per_second(self):
        gui.select_rmrc_axis(1)
        gui.select_rmrc(50)
        self.assertAlmostEqual(gui._rmrc_v[1], 0.05)       # 50 % of 100 mm/s = 0.05 m/s
        self.assertEqual(gui._rmrc_v[0], 0.0)

    def test_angular_axis_is_in_degrees_per_second(self):
        gui.select_rmrc_axis(4)
        gui.select_rmrc(-100)
        self.assertAlmostEqual(gui._rmrc_v[4], -math.radians(30))

    def test_only_one_axis_moves_at_a_time(self):
        gui.select_rmrc_axis(0)
        gui.select_rmrc(40)
        gui.select_rmrc_axis(2)
        gui.select_rmrc(40)
        self.assertEqual(sum(1 for v in gui._rmrc_v if v), 1)

    def test_step_moves_the_arm(self):
        before = self.arm.robot.q.copy()
        gui._rmrc_v[0] = 0.05
        gui._last_jog_time = time.time() - 0.05
        with patch.object(self.controller, 'rmrc_joint_velocity',
                        return_value=np.full(6, 0.1)):
            gui.step_rmrc_jog()
        self.assertTrue(np.all(self.arm.robot.q > before))

    def test_step_does_nothing_when_velocity_is_zero(self):
        before = self.arm.robot.q.copy()
        with patch.object(self.controller, 'rmrc_joint_velocity') as rmrc:
            gui.step_rmrc_jog()
        rmrc.assert_not_called()
        np.testing.assert_allclose(self.arm.robot.q, before, atol=1e-12)

    def test_step_is_ignored_while_a_cycle_runs(self):
        before = self.arm.robot.q.copy()
        gui._rmrc_v[0] = 0.05
        gui._cycle_running = True
        with patch.object(self.controller, 'rmrc_joint_velocity') as rmrc:
            gui.step_rmrc_jog()
        rmrc.assert_not_called()
        np.testing.assert_allclose(self.arm.robot.q, before, atol=1e-12)

    def test_step_does_not_move_during_estop(self):
        before = self.arm.robot.q.copy()
        gui._rmrc_v[0] = 0.05
        gui._last_jog_time = time.time() - 0.05
        self.controller.trigger_estop()
        gui.step_rmrc_jog()
        np.testing.assert_allclose(self.arm.robot.q, before, atol=1e-12)

    def test_step_stays_inside_joint_limits(self):
        self.arm.robot.q = np.full(6, math.radians(100.2))
        gui._rmrc_v[0] = 0.05
        gui._last_jog_time = time.time() - 0.05
        with patch.object(self.controller, 'rmrc_joint_velocity',
                        return_value=np.full(6, 1.0)):
            gui.step_rmrc_jog()
        self.assertTrue(np.all(self.arm.robot.q <= math.radians(100.2) + 1e-12))

class TestJointMoveBlocked(SafetyPatchedCase):
    def setUp(self):
        super().setUp()
        self.arm = FakeJointArm()
        gui._arms[:] = [self.arm]
        gui._controllers[:] = make_controllers(1)

    def test_move_that_creates_a_collision_is_blocked(self):
        self.safety_mock.side_effect = [(SAFE_COLLISION, SAFE_SINGULARITY),
                                        (UNSAFE_COLLISION, SAFE_SINGULARITY)]
        gui.select_joint(self.arm, 2, 45)
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)
        self.assertIn('Joint move blocked', gui.status['log'][-1])

    def test_move_out_of_an_unsafe_pose_is_allowed(self):
        current = dict(UNSAFE_COLLISION, distance=0.02)
        new = dict(UNSAFE_COLLISION, distance=0.05)
        self.safety_mock.side_effect = [(current, SAFE_SINGULARITY), (new, SAFE_SINGULARITY)]
        gui.select_joint(self.arm, 2, 45)
        self.assertAlmostEqual(self.arm.robot.q[2], math.radians(45), places=9)

    def test_move_deeper_into_an_unsafe_pose_is_blocked(self):
        current = dict(UNSAFE_COLLISION, distance=0.05)
        new = dict(UNSAFE_COLLISION, distance=0.02)
        self.safety_mock.side_effect = [(current, SAFE_SINGULARITY), (new, SAFE_SINGULARITY)]
        gui.select_joint(self.arm, 2, 45)
        np.testing.assert_allclose(self.arm.robot.q, self.arm.HOME, atol=1e-12)

    def test_blocked_move_puts_the_slider_back(self):
        sliders = [FakeWidget() for _ in range(6)]
        gui._joint_sliders[:] = sliders
        self.safety_mock.side_effect = [(SAFE_COLLISION, SAFE_SINGULARITY),
                                        (UNSAFE_COLLISION, SAFE_SINGULARITY)]
        gui.select_joint(self.arm, 1, 80)
        self.assertEqual(sliders[1].value, 20)

class TestProcessImageFile(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_process_image_file_success_with_stubbed_sensing(self, ):
        fake_sensing = types.ModuleType('sensing')
        fake_sensing.shape_model_from_jpeg = lambda path, target_size: object()
        with patch.dict(sys.modules, {'sensing': fake_sensing}):
            ok = gui.process_image_file('/tmp/test_shape.jpg')
        self.assertTrue(ok)
        self.assertIn('test_shape.jpg', gui.image_models)
        self.assertTrue(gui.get_status_snapshot()['image_ready'])

    def test_process_image_file_failure_reports_not_ready(self):
        fake_sensing = types.ModuleType('sensing')

        def raise_error(path, target_size):
            raise ValueError('no contour found')

        fake_sensing.shape_model_from_jpeg = raise_error
        with patch.dict(sys.modules, {'sensing': fake_sensing}):
            ok = gui.process_image_file('/tmp/bad.jpg')
        self.assertFalse(ok)
        self.assertFalse(gui.get_status_snapshot()['image_ready'])

    def test_process_image_file_stores_strokes_when_available(self):
        fake_sensing = types.ModuleType('sensing')
        fake_sensing.shape_model_from_jpeg = lambda path, target_size: object()
        fake_sensing.strokes_from_jpeg = lambda path, target_size: [object(), object()]
        with patch.dict(sys.modules, {'sensing': fake_sensing}):
            ok = gui.process_image_file('/tmp/face.jpg')
        self.assertTrue(ok)
        self.assertEqual(len(gui.image_strokes['face.jpg']), 2)

    def test_process_image_file_without_strokes_still_succeeds(self):
        fake_sensing = types.ModuleType('sensing')
        fake_sensing.shape_model_from_jpeg = lambda path, target_size: object()
        with patch.dict(sys.modules, {'sensing': fake_sensing}):
            ok = gui.process_image_file('/tmp/plain.jpg')
        self.assertTrue(ok)
        self.assertNotIn('plain.jpg', gui.image_strokes)
    
    def test_process_image_file_logs_why_strokes_failed(self):
        fake_sensing = types.ModuleType('sensing')
        fake_sensing.shape_model_from_jpeg = lambda path, target_size: object()

        def raise_error(path, target_size):
            raise ValueError('boom')

        fake_sensing.strokes_from_jpeg = raise_error
        with patch.dict(sys.modules, {'sensing': fake_sensing}):
            ok = gui.process_image_file('/tmp/face.jpg')
        self.assertTrue(ok)                                  # the outline alone is still usable
        self.assertNotIn('face.jpg', gui.image_strokes)
        self.assertTrue(any('boom' in m for m in gui.status['log']))

class TestHandleUploadRequest(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_handle_upload_request_does_nothing_when_queue_is_empty(self):
        with patch('gui.import_image') as mock_import:
            ok = gui.handle_upload_request()
        self.assertFalse(ok)
        mock_import.assert_not_called()

    def test_handle_upload_request_imports_queued_file(self):
        gui._uploaded_files.put('/tmp/cat.png')
        with patch('gui.import_image', return_value=True) as mock_import:
            ok = gui.handle_upload_request()
        self.assertTrue(ok)
        mock_import.assert_called_once_with('/tmp/cat.png')
        self.assertTrue(gui._uploaded_files.empty())

    def test_handle_upload_request_imports_every_queued_file_in_order(self):
        gui._uploaded_files.put('/tmp/a.png')
        gui._uploaded_files.put('/tmp/b.png')
        with patch('gui.import_image', side_effect=[False, True]) as mock_import:
            ok = gui.handle_upload_request()
        self.assertTrue(ok)
        self.assertEqual([c.args[0] for c in mock_import.call_args_list],
                        ['/tmp/a.png', '/tmp/b.png'])

    def test_handle_upload_request_reports_failed_import(self):
        gui._uploaded_files.put('/tmp/bad.png')
        with patch('gui.import_image', return_value=False):
            self.assertFalse(gui.handle_upload_request())

class TestOpenUploadPage(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_open_upload_page_without_server_fails(self):
        with patch('gui.webbrowser.open') as mock_open:
            ok = gui.open_upload_page()
        self.assertFalse(ok)
        mock_open.assert_not_called()
        self.assertIn('not running', gui.status['log'][-1])

    def test_open_upload_page_opens_browser_and_logs_link(self):
        with patch.object(gui, '_upload_url', 'http://127.0.0.1:9999'), \
                patch('gui.webbrowser.open', return_value=True) as mock_open:
            ok = gui.open_upload_page()
        self.assertTrue(ok)
        mock_open.assert_called_once_with('http://127.0.0.1:9999')
        self.assertIn('http://127.0.0.1:9999', gui.status['log'][-1])

    def test_open_upload_page_reports_missing_browser(self):
        with patch.object(gui, '_upload_url', 'http://127.0.0.1:9999'), \
                patch('gui.webbrowser.open', return_value=False):
            ok = gui.open_upload_page()
        self.assertFalse(ok)
        self.assertIn('Could not open the browser', gui.status['log'][-1])

    def test_open_upload_page_reports_browser_error(self):
        with patch.object(gui, '_upload_url', 'http://127.0.0.1:9999'), \
                patch('gui.webbrowser.open', side_effect=RuntimeError('no browser')):
            ok = gui.open_upload_page()
        self.assertFalse(ok)
        self.assertIn('Could not open the browser', gui.status['log'][-1])

class TestUploadServer(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        self.upload_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.upload_dir.cleanup)
        dir_patcher = patch.object(gui, 'UPLOAD_DIR', self.upload_dir.name)
        dir_patcher.start()
        self.addCleanup(dir_patcher.stop)
        # port 0 lets the OS pick a free port, so the tests never clash with a running app
        self.url = gui.start_upload_server(port=0)
        self.addCleanup(gui.stop_upload_server)

    def _get(self, path):
        try:
            with urllib.request.urlopen(self.url + path, timeout=5) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode()

    def _post(self, name, data, header=True):
        headers = {'X-Requested-With': 'dogoodbot'} if header else {}
        req = urllib.request.Request(
            f'{self.url}/upload?name={urllib.parse.quote(name)}',
            data=data, method='POST', headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode()

    def test_start_upload_server_returns_local_url(self):
        self.assertTrue(self.url.startswith('http://127.0.0.1:'))

    def test_start_upload_server_twice_returns_same_url(self):
        self.assertEqual(gui.start_upload_server(port=0), self.url)

    def test_start_upload_server_skips_busy_port(self):
        busy_port = int(self.url.rsplit(':', 1)[1])
        gui.stop_upload_server()
        # occupy the port again, then ask for it: the next port must be used instead
        import socket
        blocker = socket.socket()
        blocker.bind(('127.0.0.1', busy_port))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        url = gui.start_upload_server(port=busy_port, tries=20)
        self.assertIsNotNone(url)
        self.assertNotEqual(url, f'http://127.0.0.1:{busy_port}')

    def test_stop_upload_server_clears_url(self):
        gui.stop_upload_server()
        self.assertIsNone(gui._upload_url)

    def test_get_root_serves_the_upload_page(self):
        status, body = self._get('/')
        self.assertEqual(status, 200)
        self.assertIn('<input type="file"', body)

    def test_get_unknown_path_is_404(self):
        status, _ = self._get('/nothing')
        self.assertEqual(status, 404)

    def test_post_saves_file_and_queues_it(self):
        status, _ = self._post('cat.png', b'image-bytes')
        self.assertEqual(status, 200)
        dest = os.path.join(self.upload_dir.name, 'cat.png')
        with open(dest, 'rb') as f:
            self.assertEqual(f.read(), b'image-bytes')
        self.assertEqual(gui._uploaded_files.get_nowait(), dest)
        self.assertFalse(os.path.exists(dest + '.part'))
        self.assertIn('received by the upload page', gui.status['log'][-1])

    def test_post_strips_directories_from_the_file_name(self):
        for name in ['../../evil.png', '..\\evil2.png']:
            status, _ = self._post(name, b'x')
            self.assertEqual(status, 200)
            queued = gui._uploaded_files.get_nowait()
            self.assertEqual(os.path.dirname(queued), self.upload_dir.name)

    def test_post_rejects_unsupported_extension(self):
        status, _ = self._post('notes.txt', b'x')
        self.assertEqual(status, 400)
        self.assertTrue(gui._uploaded_files.empty())
        self.assertFalse(os.path.exists(os.path.join(self.upload_dir.name, 'notes.txt')))

    def test_post_rejects_request_without_custom_header(self):
        status, _ = self._post('cat.png', b'x', header=False)
        self.assertEqual(status, 403)
        self.assertTrue(gui._uploaded_files.empty())

    def test_post_rejects_empty_file(self):
        status, _ = self._post('cat.png', b'')
        self.assertEqual(status, 400)
        self.assertTrue(gui._uploaded_files.empty())

    def test_post_rejects_oversized_file(self):
        with patch.object(gui, 'MAX_UPLOAD_BYTES', 10):
            status, _ = self._post('cat.png', b'x' * 20)
        self.assertEqual(status, 413)
        self.assertTrue(gui._uploaded_files.empty())

    def test_post_to_unknown_path_is_404(self):
        req = urllib.request.Request(self.url + '/other', data=b'x', method='POST',
                                    headers={'X-Requested-With': 'dogoodbot'})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 404)

class TestImportImage(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        self.src_dir = tempfile.TemporaryDirectory()
        self.upload_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.src_dir.cleanup)
        self.addCleanup(self.upload_dir.cleanup)
        dir_patcher = patch.object(gui, 'UPLOAD_DIR', self.upload_dir.name)
        dir_patcher.start()
        self.addCleanup(dir_patcher.stop)
        self.fake_sensing = types.ModuleType('sensing')
        self.fake_sensing.shape_model_from_jpeg = lambda path, target_size: object()
        sensing_patcher = patch.dict(sys.modules, {'sensing': self.fake_sensing})
        sensing_patcher.start()
        self.addCleanup(sensing_patcher.stop)

    def _write(self, folder, name):
        path = os.path.join(folder, name)
        with open(path, 'wb') as f:
            f.write(b'x')
        return path

    def test_import_image_copies_file_into_upload_dir(self):
        src = self._write(self.src_dir.name, 'cat.png')
        ok = gui.import_image(src)
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(os.path.join(self.upload_dir.name, 'cat.png')))
        self.assertIn('cat.png', gui.image_models)
        snap = gui.get_status_snapshot()
        self.assertEqual(snap['image_name'], 'cat.png')
        self.assertTrue(snap['image_ready'])

    def test_import_image_keeps_the_original_file(self):
        src = self._write(self.src_dir.name, 'cat.png')
        gui.import_image(src)
        self.assertTrue(os.path.exists(src))

    def test_import_image_already_in_upload_dir_is_not_copied_onto_itself(self):
        src = self._write(self.upload_dir.name, 'dog.png')
        ok = gui.import_image(src)
        self.assertTrue(ok)
        self.assertIn('dog.png', gui.image_models)

    def test_import_image_missing_file_reports_not_ready(self):
        ok = gui.import_image(os.path.join(self.src_dir.name, 'missing.png'))
        self.assertFalse(ok)
        self.assertFalse(gui.get_status_snapshot()['image_ready'])
        self.assertIn('copy failed', gui.status['log'][-1])

    def test_import_image_processing_failure_reports_not_ready(self):
        def raise_error(path, target_size):
            raise ValueError('no contour found')

        self.fake_sensing.shape_model_from_jpeg = raise_error
        src = self._write(self.src_dir.name, 'bad.png')
        ok = gui.import_image(src)
        self.assertFalse(ok)
        self.assertFalse(gui.get_status_snapshot()['image_ready'])

class TestBuildControls(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    @patch('gui.swift')
    def test_build_controls_creates_expected_widgets(self, mock_swift):
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)

        env = FakeEnv()
        widgets = gui.build_controls(env)

        for key in ['mode_select', 'shape_select', 'image_label', 'upload_button',
                    'r_slider', 'g_slider', 'b_slider',
                    'run_button', 'stop_button', 'reset_process_button',
                    'person_select', 'curtain_label', 'crate_button',
                    'status_label', 'safety_label']:
            self.assertIn(key, widgets)
        # 23 widgets without arms: 6 section labels + task controls + RGB sliders
        # + E-STOP buttons + safety test widgets + status label + 3 event labels
        self.assertEqual(len(env.added), 23)                                        # CHANGED

    @patch('gui.swift')
    def test_image_widgets_sit_between_shape_and_colour(self, mock_swift):
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)

        env = FakeEnv()
        widgets = gui.build_controls(env)

        order = [env.added.index(widgets[key]) for key in
                ['shape_select', 'image_label', 'upload_button', 'r_slider']]       # CHANGED
        self.assertEqual(order, sorted(order))

    @patch('gui.swift')
    def test_run_stop_and_reset_process_buttons_are_in_that_order(self, mock_swift):
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)

        env = FakeEnv()
        widgets = gui.build_controls(env)

        order = [env.added.index(widgets[key]) for key in
                ['run_button', 'stop_button', 'reset_process_button', 'status_label']]
        self.assertEqual(order, sorted(order))

    @patch('gui.swift')
    def test_image_label_starts_with_no_image_selected(self, mock_swift):
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)

        widgets = gui.build_controls(FakeEnv())
        self.assertIn('No image selected', widgets['image_label'].desc)

    @patch('gui.swift')
    def test_upload_button_callback_opens_upload_page(self, mock_swift):
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)

        widgets = gui.build_controls(FakeEnv())
        with patch('gui.open_upload_page') as mock_open:
            widgets['upload_button'].kwargs['cb']()
        mock_open.assert_called_once_with()

    @patch('gui.swift')
    def test_stop_button_callback_triggers_estop(self, mock_swift):
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)

        widgets = gui.build_controls(FakeEnv(), [FakeJointArm() for _ in range(3)])
        widgets['stop_button'].kwargs['cb']()
        self.assertTrue(all(c.is_estopped() for c in gui.get_controllers()))
        self.assertEqual(gui.get_status_snapshot()['process'], 'E-STOP')

    @patch('gui.swift')
    def test_reset_process_button_callback_resets_the_process(self, mock_swift):
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)

        widgets = gui.build_controls(FakeEnv())
        with patch('gui.reset_process') as mock_reset:
            widgets['reset_process_button'].kwargs['cb']()
        mock_reset.assert_called_once_with()

    @patch('gui.swift')
    def test_mode_select_callback_updates_selection(self, mock_swift):           # CHANGED (name)
        captured_callbacks = []

        # Parameter names must match the keyword arguments used in gui.build_controls
        def fake_select(cb, desc='', options=None, value=0):                      # CHANGED (value added)
            captured_callbacks.append(cb)
            return FakeWidget(desc=desc)

        mock_swift.Select.side_effect = fake_select                               # CHANGED
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)

        env = FakeEnv()
        gui.build_controls(env)
        # build_controls creates mode_select first, so index 0 is its callback
        mode_callback = captured_callbacks[0]
        mode_callback(FakeEvent(number=1))  # index 1 -> 'basic'
        self.assertEqual(gui._selected_mode, 'basic')

    @patch('gui.swift')
    def test_slider_callbacks_update_rgb_colour(self, mock_swift):
        captured_callbacks = []

        # Parameter names must match the keyword arguments used in gui.build_controls
        def fake_slider(cb, min=0, max=255, step=1, value=0, desc=''):
            captured_callbacks.append(cb)
            return FakeWidget(desc=desc)

        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED
        mock_swift.Slider.side_effect = fake_slider
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)

        env = FakeEnv()
        gui.build_controls(env)
        # without arms, build_controls creates only the R, G, B sliders, in that order
        self.assertEqual(len(captured_callbacks), 3)
        captured_callbacks[1](100)  # G channel
        self.assertEqual(gui._selected_colour, (200, 100, 40))

class TestBuildJointControls(SafetyPatchedCase):
    def setUp(self):
        super().setUp()
        self.arms = [FakeJointArm(), FakeJointArm(), FakeJointArm()]

    def _build(self, mock_swift, arms='default'):
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        env = FakeEnv()
        arms = self.arms if arms == 'default' else arms
        widgets = gui.build_controls(env, arms)
        return env, widgets

    @patch('gui.swift')
    def test_build_controls_creates_arm_select_6_joint_sliders_reset_speed_and_rmrc(self, mock_swift):
        env, widgets = self._build(mock_swift)
        self.assertIn('arm_select', widgets)
        for j in range(1, 7):
            self.assertIn(f'j{j}_slider', widgets)
        for key in ['reset_button', 'adjust_label', 'speed_slider', 'rmrc_axis_select', 'rmrc_slider']:
            self.assertIn(key, widgets)
        # 23 base widgets + arm select + 6 joints + reset arms + adjust label + speed + rmrc select + rmrc slider
        self.assertEqual(len(env.added), 23 + 1 + 6 + 1 + 1 + 3)

    @patch('gui.swift')
    def test_no_joint_widgets_without_arms(self, mock_swift):
        env, widgets = self._build(mock_swift, arms=None)
        for key in ['arm_select', 'reset_button', 'j1_slider', 'adjust_label',
                    'speed_slider', 'rmrc_axis_select', 'rmrc_slider']:
            self.assertNotIn(key, widgets)
        # Only the 23 base widgets are created when there are no arms
        self.assertEqual(len(env.added), 23)

    @patch('gui.swift')
    def test_arm_select_lists_the_three_arms(self, mock_swift):                   # CHANGED (name)
        _, widgets = self._build(mock_swift)
        self.assertEqual(widgets['arm_select'].kwargs['options'], ['Kawasaki', 'Staubli', 'Nachi'])   # CHANGED

    @patch('gui.swift')
    def test_rmrc_axis_select_lists_the_six_axes(self, mock_swift):               # CHANGED (name)
        _, widgets = self._build(mock_swift)
        self.assertEqual(widgets['rmrc_axis_select'].kwargs['options'], gui.RMRC_LABELS)              # CHANGED

    @patch('gui.swift')
    def test_adjust_label_starts_with_the_first_robot(self, mock_swift):
        _, widgets = self._build(mock_swift)
        self.assertEqual(widgets['adjust_label'].desc, 'Adjusting: Kawasaki')

    @patch('gui.swift')
    def test_adjust_label_follows_the_arm_select(self, mock_swift):               # CHANGED (name)
        _, widgets = self._build(mock_swift)
        widgets['arm_select'].kwargs['cb'](FakeEvent(number=2))                   # CHANGED
        self.assertEqual(widgets['adjust_label'].desc, 'Adjusting: Nachi')
        widgets['arm_select'].kwargs['cb'](FakeEvent(number=1))                   # CHANGED
        self.assertEqual(widgets['adjust_label'].desc, 'Adjusting: Staubli')

    @patch('gui.swift')
    def test_one_controller_per_arm_is_created(self, mock_swift):
        self._build(mock_swift)
        self.assertEqual(len(gui.get_controllers()), 3)

    @patch('gui.swift')
    def test_given_controllers_are_used(self, mock_swift):
        controllers = make_controllers()
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        gui.build_controls(FakeEnv(), self.arms, controllers)
        for given, used in zip(controllers, gui.get_controllers()):
            self.assertIs(given, used)

    @patch('gui.swift')
    def test_speed_slider_callback_sets_the_selected_arm_speed(self, mock_swift):
        _, widgets = self._build(mock_swift)
        widgets['speed_slider'].kwargs['cb'](40)
        self.assertAlmostEqual(gui.get_controllers()[0].manual_scale, 0.4)

    @patch('gui.swift')
    def test_rmrc_slider_callback_sets_the_velocity(self, mock_swift):
        _, widgets = self._build(mock_swift)
        widgets['rmrc_axis_select'].kwargs['cb'](FakeEvent(number=2))   # Vz      # CHANGED
        widgets['rmrc_slider'].kwargs['cb'](100)
        self.assertAlmostEqual(gui._rmrc_v[2], 0.1)

    @patch('gui.swift')
    def test_joint_sliders_use_the_joint_limits_in_degrees(self, mock_swift):
        _, widgets = self._build(mock_swift)
        kwargs = widgets['j1_slider'].kwargs
        # limits are -170.5 and 100.2 deg: rounded outwards to whole degrees
        self.assertEqual(kwargs['min'], -171)
        self.assertEqual(kwargs['max'], 101)
        self.assertEqual(kwargs['step'], 1)

    @patch('gui.swift')
    def test_joint_sliders_start_at_the_first_arm_pose(self, mock_swift):
        _, widgets = self._build(mock_swift)
        self.assertEqual(widgets['j2_slider'].kwargs['value'], 20)
        self.assertEqual(widgets['j3_slider'].kwargs['value'], 70)

    @patch('gui.swift')
    def test_joint_slider_description_names_the_joint(self, mock_swift):
        _, widgets = self._build(mock_swift)
        self.assertEqual(widgets['j3_slider'].desc, 'Joint J3 (deg)')

    @patch('gui.swift')
    def test_joint_slider_callback_moves_the_first_arm_by_default(self, mock_swift):
        _, widgets = self._build(mock_swift)
        widgets['j3_slider'].kwargs['cb'](45)
        self.assertAlmostEqual(self.arms[0].robot.q[2], math.radians(45), places=9)
        # the other arms and the other joints of the same arm are untouched
        np.testing.assert_allclose(self.arms[1].robot.q, self.arms[1].HOME, atol=1e-12)
        np.testing.assert_allclose(self.arms[2].robot.q, self.arms[2].HOME, atol=1e-12)
        self.assertAlmostEqual(self.arms[0].robot.q[1], self.arms[0].HOME[1], places=9)

    @patch('gui.swift')
    def test_arm_select_callback_switches_the_controlled_arm(self, mock_swift):   # CHANGED (name)
        _, widgets = self._build(mock_swift)
        widgets['arm_select'].kwargs['cb'](FakeEvent(number=2))  # index 2 -> 'Nachi'   # CHANGED
        self.assertEqual(gui._selected_arm, 2)
        widgets['j3_slider'].kwargs['cb'](45)
        self.assertAlmostEqual(self.arms[2].robot.q[2], math.radians(45), places=9)
        np.testing.assert_allclose(self.arms[0].robot.q, self.arms[0].HOME, atol=1e-12)
        np.testing.assert_allclose(self.arms[1].robot.q, self.arms[1].HOME, atol=1e-12)

    @patch('gui.swift')
    def test_arm_select_callback_resyncs_the_sliders(self, mock_swift):           # CHANGED (name)
        self.arms[1].robot.q = np.deg2rad([0, 55, 70, 0, 90, 0])
        _, widgets = self._build(mock_swift)
        widgets['arm_select'].kwargs['cb'](FakeEvent(number=1))                   # CHANGED
        self.assertEqual(widgets['j2_slider'].value, 55)

    @patch('gui.swift')
    def test_arm_select_callback_resyncs_the_speed_slider(self, mock_swift):      # CHANGED (name)
        _, widgets = self._build(mock_swift)
        gui.get_controllers()[1].set_manual_scale(0.25)
        widgets['arm_select'].kwargs['cb'](FakeEvent(number=1))                   # CHANGED
        self.assertEqual(widgets['speed_slider'].value, 25)

    @patch('gui.swift')
    def test_reset_button_callback_returns_arms_to_home(self, mock_swift):
        _, widgets = self._build(mock_swift)
        widgets['j1_slider'].kwargs['cb'](30)
        widgets['arm_select'].kwargs['cb'](FakeEvent(number=2))                   # CHANGED
        widgets['j6_slider'].kwargs['cb'](-60)
        widgets['reset_button'].kwargs['cb']()
        for arm in self.arms:
            np.testing.assert_allclose(arm.robot.q, arm.HOME, atol=1e-12)

    @patch('gui.swift')
    def test_reset_button_after_stop_button_does_not_clear_estop(self, mock_swift):
        _, widgets = self._build(mock_swift)
        widgets['stop_button'].kwargs['cb']()
        widgets['reset_button'].kwargs['cb']()
        self.assertTrue(gui._estopped())

    @patch('gui.swift')
    def test_reset_process_button_after_stop_button_clears_estop(self, mock_swift):
        _, widgets = self._build(mock_swift)
        widgets['stop_button'].kwargs['cb']()
        self.assertTrue(gui._estopped())
        widgets['reset_process_button'].kwargs['cb']()
        self.assertFalse(gui._estopped())
        self.assertTrue(gui.consume_reset_request())

    @patch('gui.swift')
    def test_run_button_sits_before_the_teach_pendant_controls(self, mock_swift):  # CHANGED (name)
        env, widgets = self._build(mock_swift)
        order = [env.added.index(widgets[key]) for key in                          # CHANGED (order)
                ['b_slider', 'run_button', 'arm_select', 'j1_slider', 'j6_slider',
                'reset_button', 'adjust_label', 'speed_slider',
                'rmrc_axis_select', 'rmrc_slider']]
        self.assertEqual(order, sorted(order))

    @patch('gui.swift')
    def test_build_controls_registers_arms_and_sliders(self, mock_swift):
        self._build(mock_swift)
        self.assertEqual(gui._arms, self.arms)
        self.assertEqual(len(gui._joint_sliders), 6)
        self.assertEqual(gui._selected_arm, 0)

    @patch('gui.swift')
    def test_build_controls_twice_does_not_duplicate_sliders(self, mock_swift):
        self._build(mock_swift)
        gui._selected_arm = 2
        self._build(mock_swift)
        self.assertEqual(len(gui._arms), 3)
        self.assertEqual(len(gui._joint_sliders), 6)
        self.assertEqual(len(gui.get_controllers()), 3)
        self.assertEqual(gui._selected_arm, 0)

class TestRefreshStatusLabel(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_refresh_status_label_updates_text(self):
        widgets = {'status_label': FakeWidget(desc='')}
        gui.update_status(process='RUNNING', mode='basic', shape='circle', colour='red')
        gui.refresh_status_label(widgets)
        self.assertIn('RUNNING', widgets['status_label'].desc)
        self.assertIn('circle', widgets['status_label'].desc)

    def test_refresh_status_label_does_nothing_without_label(self):
        # should not raise even if 'status_label' key is missing
        gui.refresh_status_label({})

    def test_refresh_updates_image_label_with_ready_image(self):
        widgets = {'image_label': FakeWidget(desc='')}
        gui.update_status(image_name='photo.png', image_ready=True)
        gui.refresh_status_label(widgets)
        self.assertIn('photo.png', widgets['image_label'].desc)
        self.assertIn('ready', widgets['image_label'].desc)

    def test_refresh_image_label_without_image_shows_placeholder(self):
        widgets = {'image_label': FakeWidget(desc='')}
        gui.refresh_status_label(widgets)
        self.assertIn('No image selected', widgets['image_label'].desc)

    def test_refresh_image_label_shows_upload_link_when_server_runs(self):
        widgets = {'image_label': FakeWidget(desc='')}
        with patch.object(gui, '_upload_url', 'http://127.0.0.1:8765'):
            gui.refresh_status_label(widgets)
        self.assertIn('http://127.0.0.1:8765', widgets['image_label'].desc)

    def test_refresh_status_label_shows_servo_state(self):
        widgets = {'status_label': FakeWidget(desc='')}
        gui.update_status(servo='corrected 4.0 mm')
        gui.refresh_status_label(widgets)
        self.assertIn('Servo: corrected 4.0 mm', widgets['status_label'].desc)

    def test_refresh_status_label_hides_servo_when_unset(self):
        widgets = {'status_label': FakeWidget(desc='')}
        gui.refresh_status_label(widgets)
        self.assertNotIn('Servo', widgets['status_label'].desc)

    def test_refresh_updates_the_safety_label(self):
        widgets = {'safety_label': FakeWidget(desc='')}
        gui._arms[:] = [FakeJointArm()]
        gui._controllers[:] = make_controllers(1)
        with patch('gui.evaluate_safety', return_value=(SAFE_COLLISION, SAFE_SINGULARITY)):
            gui.refresh_status_label(widgets)
        self.assertIn('Safety (Kawasaki)', widgets['safety_label'].desc)
        self.assertIn('Collision: OK', widgets['safety_label'].desc)

    def test_refresh_updates_the_adjust_label(self):
        widgets = {'adjust_label': FakeWidget(desc='')}
        gui._arms[:] = [FakeJointArm(), FakeJointArm()]
        gui._selected_arm = 1
        gui.refresh_status_label(widgets)
        self.assertEqual(widgets['adjust_label'].desc, 'Adjusting: Staubli')

    def test_refresh_does_not_check_safety_while_a_cycle_runs(self):
        gui._arms[:] = [FakeJointArm()]
        gui._controllers[:] = make_controllers(1)
        gui._cycle_running = True
        with patch('gui.evaluate_safety') as evaluate:
            gui.refresh_status_label({})
        evaluate.assert_not_called()
    
    def test_image_label_shows_the_stroke_count(self):
        gui.image_strokes['face.png'] = [object(), object(), object()]
        gui.update_status(image_name='face.png', image_ready=True)
        self.assertIn('3 stroke(s)', gui._image_text())

    def test_image_label_without_strokes_has_no_stroke_count(self):
        gui.update_status(image_name='plain.png', image_ready=True)
        self.assertNotIn('stroke', gui._image_text())

if __name__ == "__main__":
    unittest.main()