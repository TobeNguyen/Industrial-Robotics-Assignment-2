# Including libraries
import os
import random
import tempfile
import time
import unittest
import urllib.request
from unittest.mock import patch

import cv2
import numpy as np
from spatialmath import SE3
from spatialgeometry import Cylinder, Cuboid, Mesh, Sphere

from workcell import (
    BOX, BOX_TOP, STORAGE_XY, FINISH_XY, STAND_TOP,
    Y_LOAD, Y_MARK, Y_PAINT, SHAPE_SIZE, PAINT_MARGIN,
    PAINTED,
    MarkedBox, MarkPool, make_arms, build_workcell, make_stamp_shape,
)
from staubli_tx2_60_task import ShapeSpec, StaubliShapeDraw
from safety import RobotSafetyController
import gui
import main
import sensing
import workcell

SAFE_COLLISION = {'safe': True, 'distance': 1.0, 'scale': 1.0, 'reason': 'No collision risk'}
UNSAFE_COLLISION = {'safe': False, 'distance': 0.02, 'scale': 0.0, 'reason': 'Self-collision risk'}
SAFE_SINGULARITY = {'safe': True, 'level': 'SAFE', 'sigma_min': 1.0}


class FakeEnv:
    def __init__(self):
        self.added = []
        self.removed = []
        self.step_count = 0
        self.dts = []

    def add(self, shape):
        self.added.append(shape)
        return shape

    def remove(self, shape):
        raise AssertionError('env.remove must not be used: Swift.step() crashes afterwards')

    def step(self, dt=0.02):
        self.step_count += 1
        self.dts.append(dt)

    def set_camera_pose(self, *args, **kwargs):
        pass

    def hold(self):
        pass

# Records the pose of an arm at every env.step (to prove that the arm does not move while it is held)
class RecordingEnv(FakeEnv):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.q_log = []

    def step(self, dt=0.02):
        super().step(dt)
        self.q_log.append(self.arm.robot.q.copy())

# Releases the E-STOP after a number of env.step calls and records the box position
class ReleaseEnv(FakeEnv):
    def __init__(self, release_after, controller):
        super().__init__()
        self.release_after = release_after
        self.controller = controller
        self.box = None
        self.box_y = []

    def step(self, dt=0.02):
        super().step(dt)
        if self.box is not None:
            self.box_y.append(float(self.box.pose.t[1]))
        if self.step_count == self.release_after:
            self.controller.reset_estop()

# Presses E-STOP and later RESET PROCESS at given env.step counts (stands in for the operator)
class ScriptedEnv(FakeEnv):
    def __init__(self, estop_at, reset_at):
        super().__init__()
        self.estop_at = estop_at
        self.reset_at = reset_at

    def step(self, dt=0.02):
        super().step(dt)
        if self.step_count == self.estop_at:
            gui.request_stop()
        if self.step_count == self.reset_at:
            gui.reset_process()

class FakeWidget:
    def __init__(self, *args, **kwargs):
        self.desc = kwargs.get('desc', '')
        self.args = args
        self.kwargs = kwargs

class FakeArm:
    def __init__(self, contact_point=(0.0, 0.0, 0.0), on_lid=True):
        self.contact_point = np.array(contact_point, dtype=float)
        self.on_lid = on_lid

    def lid_contact(self, lid_z, q=None, half_width=None):
        return self.contact_point, self.on_lid

class FakeMark:
    def __init__(self):
        self.T = None

class FakePool:
    def __init__(self):
        self.taken = 0

    def take(self):
        self.taken += 1
        return FakeMark()

class FakeShapeModel:
    def __init__(self, cx=0.0, cy=0.0, size=0.02, kind='complex'):
        self.cx, self.cy, self.size, self.kind = cx, cy, size, kind

    def boundary_points(self, n=360):
        t = np.linspace(0, 2 * np.pi, n)
        return [(self.cx + self.size * np.cos(a), self.cy + self.size * np.sin(a)) for a in t]

    def half_width_at(self, dy):
        return np.sqrt(max(self.size ** 2 - dy ** 2, 0.0))

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

# Replaces swift widgets by fakes while gui.build_controls runs
def build_gui(env, arms, controllers=None):
    with patch('gui.swift') as mock_swift:
        mock_swift.Select.side_effect = lambda *a, **k: FakeWidget(*a, **k)       # CHANGED (was Radio)
        mock_swift.Slider.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Button.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        mock_swift.Label.side_effect = lambda *a, **k: FakeWidget(*a, **k)
        return gui.build_controls(env, arms, controllers)

class TestMakeStampShape(unittest.TestCase):
    def test_circle_stamp_radius_matches_shape_size(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.025, kind='circle')
        item = make_stamp_shape(shape)
        self.assertIsInstance(item, Cylinder)
        self.assertAlmostEqual(item.radius, shape.size, places=9)

    def test_square_stamp_bounding_box_matches_shape_size(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='square')
        item = make_stamp_shape(shape)
        self.assertIsInstance(item, Cuboid)
        np.testing.assert_allclose(item.scale[:2], [2 * shape.size, 2 * shape.size])

    def test_triangle_stamp_is_a_mesh_not_a_cuboid(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='triangle')
        item = make_stamp_shape(shape)
        self.assertIsInstance(item, Mesh)
        self.assertNotIsInstance(item, Cuboid)

    def test_larger_shape_produces_larger_stamp(self):
        small = make_stamp_shape(ShapeSpec(cx=0.0, cy=0.0, size=0.01, kind='circle'))
        big = make_stamp_shape(ShapeSpec(cx=0.0, cy=0.0, size=0.03, kind='circle'))
        self.assertGreater(big.radius, small.radius)

    def test_unknown_kind_raises(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='hexagon')
        with self.assertRaises(ValueError):
            make_stamp_shape(shape)

    def test_default_color_is_painted(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='circle')
        item = make_stamp_shape(shape)
        self.assertEqual(tuple(item.color), tuple(PAINTED))

    def test_custom_color_applied(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='circle')
        red = (0.8, 0.15, 0.15, 1)
        item = make_stamp_shape(shape, color=red)
        self.assertEqual(tuple(item.color), red)

    def test_complex_stamp_is_a_mesh_with_requested_color(self):
        model = sensing.create_shape_model(
            [(-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02), (-0.02, -0.02)],
            cx=0.0, cy=0.0, size=0.02)
        green = (0.2, 0.65, 0.3, 1)
        item = make_stamp_shape(model, color=green)
        self.assertIsInstance(item, Mesh)
        self.assertEqual(tuple(item.color), green)

class TestResolveColor(unittest.TestCase):
    def test_known_colour_names_map_correctly(self):
        for name in ['red', 'blue', 'green', 'yellow']:
            self.assertEqual(main.resolve_color(name), main.COLOR_MAP[name])

    def test_unknown_colour_falls_back_to_painted(self):
        self.assertEqual(main.resolve_color('purple'), PAINTED)

    def test_none_colour_falls_back_to_painted(self):
        self.assertEqual(main.resolve_color(None), PAINTED)

    def test_rgb_tuple_is_normalised_to_rgba(self):
        self.assertEqual(main.resolve_color((255, 0, 0)), (1.0, 0.0, 0.0, 1))

    def test_rgb_list_is_accepted(self):
        self.assertEqual(main.resolve_color([0, 255, 0]), (0.0, 1.0, 0.0, 1))

    def test_rgb_values_are_clamped_to_valid_range(self):
        self.assertEqual(main.resolve_color((300, -20, 255)), (1.0, 0.0, 1.0, 1))

    def test_rgb_midpoint_is_scaled(self):
        r, g, b, a = main.resolve_color((0, 51, 255))
        self.assertAlmostEqual(g, 0.2, places=6)
        self.assertEqual(a, 1)

class TestTriangulatePolygon(unittest.TestCase):
    def _area(self, outline, triangles):
        total = 0.0
        for a, b, c in triangles:
            (ax, ay), (bx, by), (cx, cy) = outline[a], outline[b], outline[c]
            total += 0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        return total

    def test_concave_polygon_area_is_preserved(self):
        # L shape (area 3), closed loop as produced by sensing.normalise_path
        points = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 0)]
        outline, triangles = workcell._triangulate_polygon(points)
        self.assertEqual(len(triangles), len(outline) - 2)
        self.assertAlmostEqual(self._area(outline, triangles), 3.0, places=9)

    def test_clockwise_input_gives_the_same_area(self):
        points = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 0)][::-1]
        outline, triangles = workcell._triangulate_polygon(points)
        self.assertAlmostEqual(self._area(outline, triangles), 3.0, places=9)

    def test_degenerate_polygon_raises(self):
        with self.assertRaises(ValueError):
            workcell._triangulate_polygon([(0, 0), (1, 0), (2, 0), (0, 0)])
        with self.assertRaises(ValueError):
            workcell._triangulate_polygon([(0, 0), (1, 1)])

# MarkPool.reset, MarkedBox.reset and main.reset_scene (what RESET PROCESS does to the objects of the cell)
class TestSceneReset(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()

    def _pool(self, count=5):
        return MarkPool(self.env, count, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))

    def test_pool_reset_makes_every_mark_available_again(self):
        pool = self._pool()
        first = pool.take()
        pool.take()
        pool.reset()
        self.assertEqual(pool.used, 0)
        self.assertIs(pool.take(), first)

    def test_pool_reset_parks_the_marks_that_were_used(self):
        pool = self._pool()
        item = pool.take()
        item.T = SE3(0.1, 0.2, 0.8).A
        pool.reset()
        self.assertAlmostEqual(item.T[2, 3], -1.0)

    def test_pool_reset_of_an_unused_pool_does_nothing(self):
        pool = self._pool()
        pool.reset()
        self.assertEqual(pool.used, 0)

    def test_pool_that_was_exhausted_works_again_after_reset(self):
        pool = self._pool(2)
        pool.take()
        pool.take()
        self.assertIsNone(pool.take())
        pool.reset()
        self.assertIsNotNone(pool.take())

    def test_box_reset_puts_the_box_back_on_its_stand(self):
        box = MarkedBox(self.env, *STORAGE_XY, STAND_TOP)
        box.set_pose(SE3(0.0, Y_PAINT, BOX_TOP - BOX[2] / 2))
        box.reset(*STORAGE_XY, STAND_TOP)
        np.testing.assert_allclose(box.pose.t, [STORAGE_XY[0], STORAGE_XY[1], STAND_TOP + BOX[2] / 2],
                                atol=1e-9)

    def test_box_reset_forgets_the_marks(self):
        box = MarkedBox(self.env, *STORAGE_XY, STAND_TOP)
        pool = self._pool()
        box.add_mark(pool.take(), (STORAGE_XY[0], STORAGE_XY[1], BOX_TOP))
        box.reset(*STORAGE_XY, STAND_TOP, pool=pool)
        self.assertEqual(box.marks, [])

    def test_box_reset_hides_the_stamps_under_the_floor(self):
        box = MarkedBox(self.env, 0.0, Y_PAINT, STAND_TOP)
        stamp = Cylinder(radius=0.01, length=0.001, color=PAINTED)
        box.add_mark(stamp, (0.0, Y_PAINT, BOX_TOP))
        box.reset(*STORAGE_XY, STAND_TOP, pool=self._pool())
        self.assertAlmostEqual(stamp.T[2, 3], -1.0)

    def test_box_reset_parks_the_ink_marks(self):
        box = MarkedBox(self.env, 0.0, Y_MARK, STAND_TOP)
        pool = self._pool()
        ink = pool.take()
        box.add_mark(ink, (0.0, Y_MARK, BOX_TOP))
        box.reset(*STORAGE_XY, STAND_TOP, pool=pool)
        self.assertAlmostEqual(ink.T[2, 3], -1.0)

    def test_reset_scene_resets_the_box_and_the_pool(self):
        box = MarkedBox(self.env, *STORAGE_XY, STAND_TOP)
        pool = self._pool()
        box.set_pose(SE3(0.0, Y_MARK, BOX_TOP - BOX[2] / 2))
        box.add_mark(pool.take(), (0.0, Y_MARK, BOX_TOP))
        main.reset_scene(box, pool)
        self.assertEqual(pool.used, 0)
        self.assertEqual(box.marks, [])
        np.testing.assert_allclose(box.pose.t[:2], STORAGE_XY, atol=1e-9)

class TestConvey(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_convey_moves_box_to_target_y(self):
        env = FakeEnv()
        box = MarkedBox(env, 0.0, Y_LOAD, STAND_TOP)
        main.convey(env, box, Y_LOAD, Y_MARK, steps=20, dt=0.001)
        self.assertAlmostEqual(box.pose.t[1], Y_MARK, delta=1e-6)

    def test_convey_steps_the_environment(self):
        env = FakeEnv()
        box = MarkedBox(env, 0.0, Y_LOAD, STAND_TOP)
        main.convey(env, box, Y_LOAD, Y_MARK, steps=15, dt=0.001)
        self.assertEqual(env.step_count, 15)

    def test_convey_raises_on_abort(self):
        env = FakeEnv()
        box = MarkedBox(env, 0.0, Y_LOAD, STAND_TOP)
        gui._abort_requested = True
        with self.assertRaises(gui.CycleAborted):
            main.convey(env, box, Y_LOAD, Y_MARK, steps=5, dt=0.001)

    def test_convey_waits_while_estop_is_active(self):
        controller = RobotSafetyController()
        gui._controllers[:] = [controller]
        controller.trigger_estop()
        env = ReleaseEnv(release_after=3, controller=controller)
        box = MarkedBox(env, 0.0, Y_LOAD, STAND_TOP)
        env.box = box
        main.convey(env, box, Y_LOAD, Y_MARK, steps=5, dt=0.001)
        # 3 held steps, then the 5 belt steps
        self.assertEqual(env.step_count, 3 + 5)
        self.assertEqual(len(set(env.box_y[:3])), 1)           # box did not move while held
        self.assertAlmostEqual(box.pose.t[1], Y_MARK, delta=1e-6)

class TestMakeMarker(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.box = MarkedBox(self.env, 0.0, Y_MARK, STAND_TOP)

    def test_marks_land_when_on_lid(self):
        arm = FakeArm(contact_point=(0.0, Y_MARK, BOX_TOP), on_lid=True)
        pool = FakePool()
        on_frame = main.make_marker(self.env, self.box, pool, every=1, lid_z=BOX_TOP, z_lift=0.001)
        on_frame(arm, k=0)
        self.assertEqual(pool.taken, 1)
        self.assertEqual(len(self.box.marks), 1)

    def test_no_mark_when_off_lid(self):
        arm = FakeArm(contact_point=(5.0, 5.0, BOX_TOP), on_lid=False)
        pool = FakePool()
        on_frame = main.make_marker(self.env, self.box, pool, every=1, lid_z=BOX_TOP, z_lift=0.001)
        on_frame(arm, k=0)
        self.assertEqual(pool.taken, 0)

    def test_respects_every_parameter(self):
        arm = FakeArm(contact_point=(0.0, Y_MARK, BOX_TOP), on_lid=True)
        pool = FakePool()
        on_frame = main.make_marker(self.env, self.box, pool, every=3, lid_z=BOX_TOP, z_lift=0.001)
        for k in range(9):
            on_frame(arm, k)
        self.assertEqual(pool.taken, 3)

    def test_pool_exhaustion_adds_no_mark(self):
        class EmptyPool:
            def take(self):
                return None

        arm = FakeArm(contact_point=(0.0, Y_MARK, BOX_TOP), on_lid=True)
        on_frame = main.make_marker(self.env, self.box, EmptyPool(), every=1,
                                    lid_z=BOX_TOP, z_lift=0.001)
        on_frame(arm, k=0)
        self.assertEqual(len(self.box.marks), 0)

    def _lay(self, xs, spacing):
        arm = FakeArm(on_lid=True)
        pool = FakePool()
        on_frame = main.make_marker(self.env, self.box, pool, every=1, lid_z=BOX_TOP,
                                    z_lift=0.001, min_spacing=spacing)
        for k, x in enumerate(xs):
            arm.contact_point = np.array([x, Y_MARK, BOX_TOP])
            on_frame(arm, k)
        return pool.taken

    def test_min_spacing_skips_dots_that_are_too_close(self):
        self.assertEqual(self._lay([0.001 * k for k in range(10)], 0.0025), 4)

    def test_min_spacing_keeps_a_dot_far_from_the_last_one(self):
        self.assertEqual(self._lay([0.0, 0.05], 0.0025), 2)         # e.g. next stroke after a pen lift

    def test_zero_spacing_keeps_every_dot(self):
        self.assertEqual(self._lay([0.0] * 5, 0.0), 5)

    def test_skipped_dots_do_not_use_the_pool(self):
        self.assertEqual(self._lay([0.0, 0.0001, 0.0002], 0.0025), 1)

class TestRunStamp(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        self.env = FakeEnv()
        _, _, self.arm3 = make_arms()
        self.box = MarkedBox(self.env, 0.0, Y_PAINT, STAND_TOP)

    def test_run_stamp_places_exactly_one_mark(self):
        shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=0.02, kind='circle')
        stamped = main.run_stamp(self.env, self.arm3, self.box, shape, BOX_TOP)
        self.assertTrue(stamped)
        self.assertEqual(len(self.box.marks), 1)

    def test_run_stamp_uses_requested_color(self):
        shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=0.02, kind='circle')
        red = (0.8, 0.15, 0.15, 1)
        main.run_stamp(self.env, self.arm3, self.box, shape, BOX_TOP, color=red)
        placed_shape, _ = self.box.marks[0]
        self.assertEqual(tuple(placed_shape.color), red)

    def test_run_stamp_works_for_all_three_kinds(self):
        for kind in ['circle', 'square', 'triangle']:
            env = FakeEnv()
            _, _, arm3 = make_arms()
            box = MarkedBox(env, 0.0, Y_PAINT, STAND_TOP)
            shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=0.02, kind=kind)
            stamped = main.run_stamp(env, arm3, box, shape, BOX_TOP)
            self.assertTrue(stamped, f'stamp failed for kind={kind}')

    def test_run_stamp_false_for_unreachable_shape(self):
        shape = ShapeSpec(cx=10.0, cy=10.0, size=0.02, kind='circle')
        with self.assertRaises(ValueError):
            main.run_stamp(self.env, self.arm3, self.box, shape, 10.0)

    def test_run_stamp_places_a_mark_for_an_image_shape(self):
        model = sensing.create_shape_model(
            [(-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02), (-0.02, -0.02)],
            cx=0.0, cy=Y_PAINT, size=0.02)
        stamped = main.run_stamp(self.env, self.arm3, self.box, model, BOX_TOP)
        self.assertTrue(stamped)
        self.assertEqual(len(self.box.marks), 1)

class TestRunPickAndPlace(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        self.env = FakeEnv()
        self.arm1, self.arm2, self.arm3 = make_arms()
        self.box = MarkedBox(self.env, *STORAGE_XY, STAND_TOP)

    def test_run_pick_returns_a_grasp_transform(self):
        grasp = main.run_pick(self.env, self.arm1, self.box, *STORAGE_XY, STAND_TOP + BOX[2])
        self.assertIsInstance(grasp, SE3)

    def test_box_ends_at_place_target(self):
        grasp = main.run_pick(self.env, self.arm1, self.box, *STORAGE_XY, STAND_TOP + BOX[2])
        main.run_place(self.env, self.arm1, self.box, grasp, 0.0, Y_LOAD, BOX_TOP)
        self.assertAlmostEqual(self.box.pose.t[1], Y_LOAD, delta=0.001)

class TestRunCycle(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def _fresh(self):
        env = FakeEnv()
        build_workcell(env)
        arms = make_arms()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        ink_pool = MarkPool(env, 140, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        return env, arms, box, ink_pool

    def test_full_cycle_completes_and_marks_both_stations(self):
        env, arms, box, ink_pool = self._fresh()
        main.run_cycle(env, arms, box, ink_pool)
        self.assertGreater(ink_pool.used, 0)
        self.assertEqual(len(box.marks) - ink_pool.used, 1)

    def test_run_cycle_uses_requested_color(self):
        env, arms, box, ink_pool = self._fresh()
        blue = main.COLOR_MAP['blue']
        main.run_cycle(env, arms, box, ink_pool, color=blue)
        stamp_shape = box.marks[-1][0]
        self.assertEqual(tuple(stamp_shape.color), blue)

class TestCheckLayout(unittest.TestCase):
    def test_check_layout_passes_for_default_stations(self):
        self.assertTrue(main.check_layout())

class TestBuildDrawShape(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_random_mode_returns_shape_spec_at_given_position(self):
        shape = main.build_draw_shape({'mode': 'random'}, cx=0.1, cy=0.2, size=0.03)
        self.assertIsInstance(shape, ShapeSpec)
        self.assertEqual(shape.cx, 0.1)

    def test_basic_mode_without_shape_raises(self):
        with self.assertRaises(ValueError):
            main.build_draw_shape({'mode': 'basic', 'shape': None}, cx=0.0, cy=0.0)

    def test_image_mode_returns_stashed_model_repositioned(self):
        model = FakeShapeModel(cx=5.0, cy=5.0, size=0.025, kind='complex')
        gui.image_models['cat.jpg'] = model
        gui.update_status(image_name='cat.jpg', image_ready=True)
        shape = main.build_draw_shape({'mode': 'image'}, cx=0.0, cy=Y_MARK)
        self.assertIs(shape, model)
        self.assertEqual(shape.cy, Y_MARK)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            main.build_draw_shape({'mode': 'bogus'}, cx=0.0, cy=0.0)

class TestImageImportFlow(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        self.src_dir = tempfile.TemporaryDirectory()
        self.upload_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.src_dir.cleanup)
        self.addCleanup(self.upload_dir.cleanup)
        dir_patcher = patch.object(gui, 'UPLOAD_DIR', self.upload_dir.name)
        dir_patcher.start()
        self.addCleanup(dir_patcher.stop)

    def _square_png(self, name='square.png'):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[60:140, 60:140] = 255
        path = os.path.join(self.src_dir.name, name)
        cv2.imwrite(path, img)
        return path

    def test_imported_image_becomes_the_draw_shape(self):
        self.assertTrue(gui.import_image(self._square_png()))
        shape = main.build_draw_shape({'mode': 'image'}, cx=0.0, cy=Y_MARK)
        self.assertIs(shape, gui.image_models['square.png'])
        self.assertEqual(shape.cy, Y_MARK)

    def test_imported_image_allows_image_mode_run(self):
        gui.import_image(self._square_png())
        gui.select_mode('image')
        self.assertTrue(gui.request_run())
        command = gui.get_pending_command()
        self.assertEqual(command['mode'], 'image')

    def test_page_upload_flows_through_to_the_draw_shape(self):
        # port 0 lets the OS pick a free port, so the test never clashes with a running app
        url = gui.start_upload_server(port=0)
        self.addCleanup(gui.stop_upload_server)
        with open(self._square_png('flow.png'), 'rb') as f:
            data = f.read()
        request = urllib.request.Request(
            f'{url}/upload?name=flow.png', data=data, method='POST',
            headers={'X-Requested-With': 'dogoodbot'})
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)

        ok = gui.handle_upload_request()
        self.assertTrue(ok)
        snap = gui.get_status_snapshot()
        self.assertEqual(snap['image_name'], 'flow.png')
        self.assertTrue(snap['image_ready'])
        shape = main.build_draw_shape({'mode': 'image'}, cx=0.0, cy=Y_MARK)
        self.assertIs(shape, gui.image_models['flow.png'])

class TestVisualServoing(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        self.env = FakeEnv()
        self.box = MarkedBox(self.env, 0.0, Y_MARK, STAND_TOP)
        self.box.set_pose(SE3(0.0, Y_MARK, self.box.pose.t[2]))
        # wrist camera pose with the Staubli at its look pose above the lid
        arm = make_arms()[1]
        arm.robot.q = arm.plan_look(0.0, Y_MARK, BOX_TOP, main.CAM_HEIGHT)[0][1][-1]
        self.cam = arm.cam_pose()

    def _square(self, size, cx=0.0, cy=Y_MARK):
        pts = [(-size, -size), (size, -size), (size, size), (-size, size), (-size, -size)]
        return sensing.create_shape_model(pts, cx=cx, cy=cy, size=size)

    def test_simulated_camera_frames_are_ready(self):
        rgb, depth = main.simulate_rgbd(self.box)
        self.assertTrue(sensing.get_sensor_status(rgb, depth)['ready'])

    def test_corrected_stroke_moves_centre_and_keeps_shape(self):
        stroke = self._square(0.02)
        out = main.corrected_stroke(stroke, SE3(0.01, 0.02, 0.0))
        self.assertAlmostEqual(out.cx, stroke.cx + 0.01, places=9)
        self.assertAlmostEqual(out.cy, stroke.cy + 0.02, places=9)
        np.testing.assert_allclose(out.points, stroke.points, atol=1e-9)

    def test_corrected_stroke_applies_rotation(self):
        stroke = sensing.create_shape_model([(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)],
                                            cx=0.0, cy=0.0, size=1.0)
        out = main.corrected_stroke(stroke, SE3.Rz(np.pi / 2))
        np.testing.assert_allclose(out.points[0], [0.0, 1.0], atol=1e-9)

    def test_get_draw_strokes_only_for_image_mode(self):
        self.assertIsNone(main.get_draw_strokes({'mode': 'basic'}, cx=0.0, cy=Y_MARK))

    def test_get_draw_strokes_returns_repositioned_strokes(self):
        strokes = [self._square(0.02, cx=5.0, cy=5.0), self._square(0.01, cx=5.0, cy=5.0)]
        gui.image_strokes['face.png'] = strokes
        gui.update_status(image_name='face.png', image_ready=True)
        out = main.get_draw_strokes({'mode': 'image'}, cx=0.0, cy=Y_MARK)
        self.assertIs(out, strokes)
        self.assertTrue(all(s.cy == Y_MARK for s in out))

    def test_get_draw_strokes_none_without_stored_strokes(self):
        gui.update_status(image_name='plain.png', image_ready=True)
        self.assertIsNone(main.get_draw_strokes({'mode': 'image'}, cx=0.0, cy=Y_MARK))

    def test_run_cycle_draws_every_stroke_and_servos(self):
        env = FakeEnv()
        build_workcell(env)
        arms = make_arms()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        ink_pool = MarkPool(env, 400, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        strokes = [self._square(0.02), self._square(0.01)]
        main.run_cycle(env, arms, box, ink_pool, shape=strokes[0], strokes=strokes)
        self.assertGreater(ink_pool.used, 0)
        self.assertEqual(len(box.marks) - ink_pool.used, 1)
        self.assertIsNotNone(gui.get_status_snapshot()['servo'])

    def test_measure_correction_is_identity_when_box_has_not_moved(self):
        reference = main.capture_reference(self.box)
        correction, state = main.measure_correction(self.box, reference)
        self.assertFalse(state['applied'])
        np.testing.assert_allclose(correction.A, np.eye(4), atol=1e-9)

    def test_measure_correction_follows_a_shifted_box(self):
        reference = main.capture_reference(self.box)
        self.box.set_pose(SE3(0.005, 0.0, 0.0) * self.box.pose)
        correction, state = main.measure_correction(self.box, reference)
        self.assertTrue(state['applied'])
        self.assertAlmostEqual(correction.t[0], 0.005, delta=0.0015)
        self.assertAlmostEqual(correction.t[1], 0.0, delta=0.0015)

    def test_servo_strokes_gives_every_stroke_the_same_shift(self):
        reference = main.capture_reference(self.box)
        strokes = [self._square(0.03), self._square(0.02), self._square(0.01)]
        out = main.servo_strokes(self.env, self.box, strokes, reference, nudge=(0.004, 0.0))
        shifts = [o.cx - s.cx for o, s in zip(out, strokes)]
        for shift in shifts:
            self.assertAlmostEqual(shift, shifts[0], places=9)
        self.assertAlmostEqual(shifts[0], 0.004, delta=0.0015)
        self.assertIn('corrected', gui.get_status_snapshot()['servo'])

    def test_servo_strokes_leaves_strokes_alone_when_on_target(self):
        reference = main.capture_reference(self.box)
        strokes = [self._square(0.03), self._square(0.02)]
        out = main.servo_strokes(self.env, self.box, strokes, reference)
        self.assertIs(out[0], strokes[0])
        self.assertIs(out[1], strokes[1])
        self.assertEqual(gui.get_status_snapshot()['servo'], 'on target')

    def test_servo_strokes_reads_the_camera_the_same_number_of_times_for_any_stroke_count(self):
        reference = main.capture_reference(self.box)
        counts = []
        for n in (2, 6):
            with patch.object(main, 'simulate_rgbd', wraps=main.simulate_rgbd) as camera:
                main.servo_strokes(self.env, self.box, [self._square(0.02)] * n, reference)
            counts.append(camera.call_count)
        self.assertEqual(counts, [main.SERVO_FRAMES, main.SERVO_FRAMES])

    def test_servo_strokes_camera_not_ready_leaves_strokes_alone(self):
        reference = main.capture_reference(self.box)
        strokes = [self._square(0.03), self._square(0.02)]
        with patch.object(main, 'simulate_rgbd', return_value=(None, np.zeros((240, 320)))):
            out = main.servo_strokes(self.env, self.box, strokes, reference)
        self.assertEqual(out, strokes)
        self.assertEqual(gui.get_status_snapshot()['servo'], 'camera not ready')

    def _multi_stroke_cycle(self, strokes):
        env = FakeEnv()
        build_workcell(env)
        arms = make_arms()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        ink_pool = MarkPool(env, 400, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        main.run_cycle(env, arms, box, ink_pool, shape=strokes[0], strokes=strokes)

    def test_run_cycle_measures_the_box_once_for_all_strokes(self):
        strokes = [self._square(0.02), self._square(0.015), self._square(0.01)]
        with patch.object(main, 'SERVO_EVERY_STROKE', False), \
                patch('main.servo_strokes', wraps=main.servo_strokes) as measure:
            self._multi_stroke_cycle(strokes)
        measure.assert_called_once()

    def test_run_cycle_measures_before_every_stroke_when_enabled(self):
        strokes = [self._square(0.02), self._square(0.015), self._square(0.01)]
        with patch.object(main, 'SERVO_EVERY_STROKE', True), \
                patch('main.servo_strokes', wraps=main.servo_strokes) as measure:
            self._multi_stroke_cycle(strokes)
        self.assertEqual(measure.call_count, 3)
        self.assertEqual(len(measure.call_args_list[0].args[2]), 3)    # first call: all strokes
        self.assertEqual(len(measure.call_args_list[1].args[2]), 1)    # later calls: one stroke each

    def test_run_cycle_goes_to_the_look_pose_before_each_stroke(self):
        strokes = [self._square(0.02), self._square(0.015), self._square(0.01)]
        with patch.object(main, 'SERVO_EVERY_STROKE', True), \
                patch.object(StaubliShapeDraw, 'plan_look', autospec=True,
                            side_effect=StaubliShapeDraw.plan_look) as look:
            self._multi_stroke_cycle(strokes)
        self.assertEqual(look.call_count, 3)

    def test_single_shape_cycle_does_not_use_the_camera(self):
        env = FakeEnv()
        build_workcell(env)
        arms = make_arms()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        ink_pool = MarkPool(env, 140, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        with patch.object(StaubliShapeDraw, 'plan_look', autospec=True) as look:
            main.run_cycle(env, arms, box, ink_pool)
        look.assert_not_called()

    def test_lid_is_centred_in_the_wrist_camera_image(self):
        rgb, depth = main.simulate_rgbd(self.box, self.cam)
        ys, xs = np.nonzero(rgb[:, :, 0])
        self.assertAlmostEqual(xs.mean(), main.CAM_INTRINSICS['cx'], delta=2.0)
        self.assertAlmostEqual(ys.mean(), main.CAM_INTRINSICS['cy'], delta=2.0)
        self.assertAlmostEqual(float(depth[0, 0]), main.CAM_HEIGHT, delta=0.002)

    def test_simulated_camera_frames_are_ready(self):
        rgb, depth = main.simulate_rgbd(self.box, self.cam)
        self.assertTrue(sensing.get_sensor_status(rgb, depth)['ready'])

    def test_measure_correction_is_identity_when_box_has_not_moved(self):
        reference = main.capture_reference(self.box, self.cam)
        correction, state = main.measure_correction(self.box, reference, self.cam)
        self.assertFalse(state['applied'])
        np.testing.assert_allclose(correction.A, np.eye(4), atol=1e-9)

    def test_measure_correction_follows_a_shifted_box(self):
        reference = main.capture_reference(self.box, self.cam)
        self.box.set_pose(SE3(0.005, 0.0, 0.0) * self.box.pose)
        correction, state = main.measure_correction(self.box, reference, self.cam)
        self.assertTrue(state['applied'])
        self.assertAlmostEqual(correction.t[0], 0.005, delta=0.0015)
        self.assertAlmostEqual(correction.t[1], 0.0, delta=0.0015)

    def test_servo_strokes_gives_every_stroke_the_same_shift(self):
        reference = main.capture_reference(self.box, self.cam)
        strokes = [self._square(0.03), self._square(0.02), self._square(0.01)]
        out = main.servo_strokes(self.env, self.box, strokes, reference, self.cam, nudge=(0.004, 0.0))
        shifts = [o.cx - s.cx for o, s in zip(out, strokes)]
        for shift in shifts:
            self.assertAlmostEqual(shift, shifts[0], places=9)
        self.assertAlmostEqual(shifts[0], 0.004, delta=0.0015)
        self.assertIn('corrected', gui.get_status_snapshot()['servo'])

    def test_servo_strokes_leaves_strokes_alone_when_on_target(self):
        reference = main.capture_reference(self.box, self.cam)
        strokes = [self._square(0.03), self._square(0.02)]
        out = main.servo_strokes(self.env, self.box, strokes, reference, self.cam)
        self.assertIs(out[0], strokes[0])
        self.assertIs(out[1], strokes[1])
        self.assertEqual(gui.get_status_snapshot()['servo'], 'on target')

    def test_servo_strokes_reads_the_camera_the_same_number_of_times_for_any_stroke_count(self):
        reference = main.capture_reference(self.box, self.cam)
        counts = []
        for n in (2, 6):
            with patch.object(main, 'simulate_rgbd', wraps=main.simulate_rgbd) as camera:
                main.servo_strokes(self.env, self.box, [self._square(0.02)] * n, reference, self.cam)
            counts.append(camera.call_count)
        self.assertEqual(counts, [main.SERVO_FRAMES, main.SERVO_FRAMES])

    def test_servo_strokes_camera_not_ready_leaves_strokes_alone(self):
        reference = main.capture_reference(self.box, self.cam)
        strokes = [self._square(0.03), self._square(0.02)]
        with patch.object(main, 'simulate_rgbd', return_value=(None, np.zeros((240, 320)))):
            out = main.servo_strokes(self.env, self.box, strokes, reference, self.cam)
        self.assertEqual(out, strokes)
        self.assertEqual(gui.get_status_snapshot()['servo'], 'camera not ready')

class TestManyStrokes(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_six_strokes_fit_in_a_pool_that_the_old_marking_would_exhaust(self):
        env = FakeEnv()
        build_workcell(env)
        arms = make_arms()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        # the old marking (every 3rd point, ~120 dots per stroke) needed 6 * 120 = 720 dots
        ink_pool = MarkPool(env, 400, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        strokes = []
        for size in (0.008, 0.010, 0.012, 0.014, 0.016, 0.018):
            pts = [(-size, -size), (size, -size), (size, size), (-size, size), (-size, -size)]
            strokes.append(sensing.create_shape_model(pts, cx=0.0, cy=Y_MARK, size=size))
        main.run_cycle(env, arms, box, ink_pool, shape=strokes[0], strokes=strokes)
        self.assertGreater(ink_pool.used, 6 * 20)
        self.assertLess(ink_pool.used, 400)
        self.assertFalse(any('Ink pool' in m for m in gui.status['log']))

    def test_run_cycle_reports_when_the_ink_pool_is_used_up(self):
        env = FakeEnv()
        build_workcell(env)
        arms = make_arms()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        ink_pool = MarkPool(env, 5, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        main.run_cycle(env, arms, box, ink_pool)
        self.assertTrue(any('Ink pool used up' in m for m in gui.status['log']))

class TestInkSettings(unittest.TestCase):
    def test_ink_dots_overlap_so_a_stroke_looks_continuous(self):
        self.assertLessEqual(workcell.INK_SPACING, 2 * workcell.INK_RADIUS)

    def test_pool_is_long_enough_for_a_busy_image(self):
        # about 2 m of line is a realistic worst case for a busy image
        self.assertGreaterEqual(workcell.INK_POOL_SIZE * workcell.INK_SPACING, 2.0)

class TestRunGuiCycle(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def _fresh(self):
        env = FakeEnv()
        build_workcell(env)
        arms = make_arms()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        ink_pool = MarkPool(env, 140, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        return env, arms, box, ink_pool

    def test_run_gui_cycle_applies_requested_colour(self):
        env, arms, box, ink_pool = self._fresh()
        command = {'mode': 'basic', 'shape': 'circle', 'colour': 'green'}
        main.run_gui_cycle(env, arms, box, ink_pool, command)
        stamp_shape = box.marks[-1][0]
        self.assertEqual(tuple(stamp_shape.color), main.COLOR_MAP['green'])

    def test_run_gui_cycle_applies_rgb_colour(self):
        env, arms, box, ink_pool = self._fresh()
        command = {'mode': 'basic', 'shape': 'circle', 'colour': (255, 0, 0)}
        main.run_gui_cycle(env, arms, box, ink_pool, command)
        stamp_shape = box.marks[-1][0]
        self.assertEqual(tuple(stamp_shape.color), (1.0, 0.0, 0.0, 1))

    def test_run_gui_cycle_missing_image_model_reports_error_without_crashing(self):
        env, arms, box, ink_pool = self._fresh()
        gui.update_status(image_name='missing.jpg', image_ready=False)
        command = {'mode': 'image', 'shape': None, 'colour': 'green'}
        ok = main.run_gui_cycle(env, arms, box, ink_pool, command)
        self.assertFalse(ok)
        self.assertEqual(gui.get_status_snapshot()['process'], 'ERROR')

    def test_run_gui_cycle_abort_returns_false_and_goes_back_to_ready(self):
        gui.update_status(process='RUNNING')
        gui._cycle_running = True
        gui._abort_requested = True
        command = {'mode': 'basic', 'shape': 'circle', 'colour': 'green'}
        with patch('main.run_cycle', side_effect=gui.CycleAborted()):
            ok = main.run_gui_cycle(FakeEnv(), None, None, None, command)
        self.assertFalse(ok)
        self.assertEqual(gui.get_status_snapshot()['process'], 'READY')
        self.assertFalse(gui._cycle_running)
        self.assertFalse(gui.abort_requested())
        self.assertIn('aborted', gui.status['log'][-1])

    def test_run_gui_cycle_passes_the_controllers_on(self):
        controllers = [RobotSafetyController(), RobotSafetyController(), RobotSafetyController()]
        command = {'mode': 'basic', 'shape': 'circle', 'colour': 'green'}
        with patch('main.run_cycle') as run_cycle:
            main.run_gui_cycle(FakeEnv(), None, None, None, command, controllers=controllers)
        self.assertIs(run_cycle.call_args.kwargs['controllers'], controllers)

# Arm Select, six joint sliders and the reset buttons wired to the three real arms from make_arms()
class TestJointControls(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        self.env = FakeEnv()
        self.arms = make_arms()
        # these tests check the widget wiring, not the collision logic
        patcher = patch('gui.evaluate_safety', return_value=(SAFE_COLLISION, SAFE_SINGULARITY))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build(self, arms=None):
        arms = self.arms if arms is None else arms
        return build_gui(self.env, arms)

    # A joint value in whole degrees that differs from home but stays inside the joint limits
    def _nudged_deg(self, arm, joint):
        home = float(np.degrees(arm.HOME[joint]))
        hi = float(np.degrees(arm.robot.qlim[1][joint]))
        return home + 10 if home + 10 <= hi else home - 10

    def test_arm_select_six_sliders_and_reset_buttons_exist(self):                # CHANGED (name)
        widgets = self._build()
        self.assertIn('arm_select', widgets)                                      # CHANGED
        for j in range(1, 7):
            self.assertIn(f'j{j}_slider', widgets)
        self.assertIn('reset_button', widgets)
        self.assertIn('reset_process_button', widgets)
        self.assertEqual(len(gui._joint_sliders), 6)

    def test_speed_and_rmrc_widgets_exist(self):
        widgets = self._build()
        for key in ['adjust_label', 'speed_slider', 'rmrc_axis_select', 'rmrc_slider', 'safety_label']:   # CHANGED
            self.assertIn(key, widgets)

    def test_adjust_label_names_the_robot_of_every_arm(self):
        widgets = self._build()
        for i, name in enumerate(gui.ARM_NAMES):
            widgets['arm_select'].kwargs['cb'](i)                                 # CHANGED
            self.assertEqual(widgets['adjust_label'].desc, f'Adjusting: {name}')

    def test_arm_select_lists_the_three_arms(self):                               # CHANGED (name)
        widgets = self._build()
        self.assertEqual(widgets['arm_select'].kwargs['options'], ['Kawasaki', 'Staubli', 'Nachi'])   # CHANGED

    def test_sliders_cover_the_selected_arms_real_joint_limits(self):
        widgets = self._build()
        for i, arm in enumerate(self.arms):
            widgets['arm_select'].kwargs['cb'](i)                                 # CHANGED
            qlim = arm.robot.qlim
            for j in range(6):
                slider = widgets[f'j{j + 1}_slider']
                self.assertLessEqual(slider.min, np.degrees(qlim[0][j]) + 1e-9)
                self.assertGreaterEqual(slider.max, np.degrees(qlim[1][j]) - 1e-9)
                self.assertLessEqual(slider.min, slider.value)
                self.assertLessEqual(slider.value, slider.max)

    def test_sliders_start_at_the_first_arms_home_pose(self):
        widgets = self._build()
        for j in range(6):
            kwargs = widgets[f'j{j + 1}_slider'].kwargs
            self.assertAlmostEqual(kwargs['value'], np.degrees(self.arms[0].HOME[j]), delta=1.0)

    def test_switching_arm_moves_the_sliders_to_that_arms_pose(self):
        widgets = self._build()
        widgets['arm_select'].kwargs['cb'](1)                                     # CHANGED
        for j in range(6):
            slider = widgets[f'j{j + 1}_slider']
            self.assertAlmostEqual(slider.value, np.degrees(self.arms[1].HOME[j]), delta=1.0)

    def test_slider_moves_the_first_arm_by_default(self):
        widgets = self._build()
        arm = self.arms[0]
        before = arm.fk().A.copy()
        deg = self._nudged_deg(arm, 0)
        widgets['j1_slider'].kwargs['cb'](deg)
        self.assertAlmostEqual(arm.robot.q[0], np.radians(deg), places=9)
        self.assertFalse(np.allclose(arm.fk().A, before))

    def test_slider_only_moves_the_selected_arm(self):
        widgets = self._build()
        homes = [arm.robot.q.copy() for arm in self.arms]
        widgets['arm_select'].kwargs['cb'](2)                                     # CHANGED
        deg = self._nudged_deg(self.arms[2], 5)
        widgets['j6_slider'].kwargs['cb'](deg)
        self.assertAlmostEqual(self.arms[2].robot.q[5], np.radians(deg), places=9)
        np.testing.assert_allclose(self.arms[0].robot.q, homes[0], atol=1e-12)
        np.testing.assert_allclose(self.arms[1].robot.q, homes[1], atol=1e-12)

    def test_slider_clamps_to_the_selected_arms_joint_limits(self):
        widgets = self._build()
        widgets['arm_select'].kwargs['cb'](1)                                     # CHANGED
        widgets['j2_slider'].kwargs['cb'](100000)
        self.assertAlmostEqual(self.arms[1].robot.q[1], self.arms[1].robot.qlim[1][1], places=9)

    def test_reset_arms_returns_all_three_arms_to_home(self):
        widgets = self._build()
        for i, arm in enumerate(self.arms):
            widgets['arm_select'].kwargs['cb'](i)                                 # CHANGED
            widgets['j1_slider'].kwargs['cb'](self._nudged_deg(arm, 0))
        widgets['reset_button'].kwargs['cb']()
        for arm in self.arms:
            np.testing.assert_allclose(arm.robot.q, arm.HOME, atol=1e-9)
            np.testing.assert_allclose(arm.fk().t, arm.fk(arm.HOME).t, atol=1e-9)

    def test_reset_arms_puts_the_sliders_back_to_home(self):
        widgets = self._build()
        widgets['j2_slider'].kwargs['cb'](self._nudged_deg(self.arms[0], 1))
        widgets['reset_button'].kwargs['cb']()
        slider = widgets['j2_slider']
        self.assertAlmostEqual(slider.value, np.degrees(self.arms[0].HOME[1]), delta=1.0)

    def test_reset_process_returns_all_three_arms_to_home(self):
        widgets = self._build()
        for i, arm in enumerate(self.arms):
            widgets['arm_select'].kwargs['cb'](i)                                 # CHANGED
            widgets['j1_slider'].kwargs['cb'](self._nudged_deg(arm, 0))
        widgets['reset_process_button'].kwargs['cb']()
        gui.finish_reset_process()                                                # ADDED
        for arm in self.arms:
            np.testing.assert_allclose(arm.robot.q, arm.HOME, atol=1e-9)

    def test_sliders_and_resets_are_ignored_while_running(self):
        widgets = self._build()
        widgets['j1_slider'].kwargs['cb'](self._nudged_deg(self.arms[0], 0))
        moved = [arm.robot.q.copy() for arm in self.arms]
        gui._cycle_running = True
        widgets['j1_slider'].kwargs['cb'](0)
        widgets['reset_button'].kwargs['cb']()
        widgets['reset_process_button'].kwargs['cb']()
        for arm, q in zip(self.arms, moved):
            np.testing.assert_allclose(arm.robot.q, q, atol=1e-12)
        self.assertFalse(gui.consume_reset_request())

    def test_stop_button_blocks_sliders_until_reset_process(self):
        widgets = self._build()
        widgets['stop_button'].kwargs['cb']()
        before = self.arms[0].robot.q.copy()
        widgets['j1_slider'].kwargs['cb'](self._nudged_deg(self.arms[0], 0))
        np.testing.assert_allclose(self.arms[0].robot.q, before, atol=1e-12)
        widgets['reset_button'].kwargs['cb']()          # RESET ARMS does not clear the E-STOP
        self.assertTrue(gui._estopped())
        widgets['reset_process_button'].kwargs['cb']()
        self.assertFalse(gui._estopped())
        gui.finish_reset_process()                                                # ADDED
        widgets['j1_slider'].kwargs['cb'](self._nudged_deg(self.arms[0], 0))
        self.assertFalse(np.allclose(self.arms[0].robot.q, before))

    def test_controls_work_again_after_a_cycle(self):
        env = FakeEnv()
        build_workcell(env)
        arms = make_arms()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        ink_pool = MarkPool(env, 140, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        widgets = self._build(arms)

        # request_run() sets RUNNING, run_gui_cycle() ends with COMPLETE
        gui.update_status(process='RUNNING')
        gui._cycle_running = True
        command = {'mode': 'basic', 'shape': 'circle', 'colour': 'green'}
        self.assertTrue(main.run_gui_cycle(env, arms, box, ink_pool, command))
        self.assertEqual(gui.get_status_snapshot()['process'], 'COMPLETE')

        deg = float(np.degrees(arms[0].HOME[0])) + 10
        widgets['j1_slider'].kwargs['cb'](deg)
        self.assertAlmostEqual(arms[0].robot.q[0], np.radians(deg), places=9)
        widgets['reset_button'].kwargs['cb']()
        for arm in arms:
            np.testing.assert_allclose(arm.robot.q, arm.HOME, atol=1e-9)

# play() with safety: converts the speed scale into a time step, holds on E-STOP, aborts on RESET PROCESS.
# How the scale itself is computed (manual speed, collision, singularity) is tested in test_safety.py,
# so gui.safety_scale is patched wherever a specific scale is needed.
class TestPlaySafety(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        self.arm, _, _ = make_arms()
        self.arm.robot.q = self.arm.HOME.copy()
        self.controller = RobotSafetyController(robot_name='Kawasaki')
        patcher = patch('gui.evaluate_safety', return_value=(SAFE_COLLISION, SAFE_SINGULARITY))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _traj(self, n=3):
        return [self.arm.HOME.copy() for _ in range(n)]

    def test_play_without_controller_uses_the_normal_time_step(self):
        env = FakeEnv()
        main.play(env, self.arm, self._traj(), dt=0.02)
        self.assertEqual(env.dts, [0.02] * 3)

    def test_play_slows_down_proportionally_to_the_scale(self):
        env = FakeEnv()
        with patch('gui.safety_scale', return_value=0.5):
            main.play(env, self.arm, self._traj(), dt=0.02, controller=self.controller)
        for dt in env.dts:
            self.assertAlmostEqual(dt, 0.04)

    def test_play_never_slows_down_more_than_the_minimum_scale(self):
        env = FakeEnv()
        with patch('gui.safety_scale', return_value=0.01):
            main.play(env, self.arm, self._traj(1), dt=0.02, controller=self.controller)
        self.assertAlmostEqual(env.dts[0], 0.02 / main.MIN_SCALE)

    def test_play_holds_position_on_estop(self):
        env = RecordingEnv(self.arm)
        start = self.arm.HOME.copy()
        target = start.copy()
        target[0] += 0.05
        with patch('gui.safety_scale', side_effect=[0.0, 0.0, 1.0]):
            main.play(env, self.arm, [target], dt=0.02, controller=self.controller)
        self.assertEqual(env.step_count, 3)                     # 2 held steps + 1 move
        np.testing.assert_allclose(env.q_log[0], start, atol=1e-12)
        np.testing.assert_allclose(env.q_log[1], start, atol=1e-12)
        np.testing.assert_allclose(env.q_log[2], target, atol=1e-12)

    def test_play_raises_on_abort(self):
        env = FakeEnv()
        gui._abort_requested = True
        with self.assertRaises(gui.CycleAborted):
            main.play(env, self.arm, self._traj(), dt=0.02, controller=self.controller)
        self.assertEqual(env.step_count, 0)

    def test_play_raises_on_abort_even_without_controller(self):
        gui._abort_requested = True
        with self.assertRaises(gui.CycleAborted):
            main.play(FakeEnv(), self.arm, self._traj(), dt=0.02)

    def test_abort_while_held_ends_the_hold(self):
        # E-STOP, then RESET PROCESS (which requests the abort) while play() is holding
        gui._controllers[:] = [self.controller]
        gui._cycle_running = True
        gui.request_stop()
        env = ScriptedEnv(estop_at=None, reset_at=3)
        with self.assertRaises(gui.CycleAborted):
            main.play(env, self.arm, self._traj(), dt=0.02, controller=self.controller)
        self.assertEqual(env.step_count, 3)

class TestHomePosesAreSafe(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_every_arm_is_safe_at_home(self):
        arms = make_arms()
        gui._arms[:] = list(arms)
        for arm in arms:
            arm.robot.q = arm.HOME.copy()
        for name, arm in zip(gui.ARM_NAMES, arms):
            collision, singularity = gui.evaluate_safety(arm)
            self.assertTrue(collision['safe'], f'{name}: {collision}')
            self.assertNotEqual(singularity['level'], 'STOP', f'{name}: {singularity}')

class TestEstopResetFlow(unittest.TestCase):
    def setUp(self):
        reset_gui_state()
        # this test checks the E-STOP / RESET PROCESS flow, not the collision logic (keeps it fast)
        patcher = patch('gui.evaluate_safety', return_value=(SAFE_COLLISION, SAFE_SINGULARITY))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_estop_then_reset_process_then_run_again(self):
        env = ScriptedEnv(estop_at=200, reset_at=210)
        build_workcell(env)
        arms = make_arms()
        build_gui(env, arms)
        controllers = gui.get_controllers()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        ink_pool = MarkPool(env, 400, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        command = {'mode': 'basic', 'shape': 'circle', 'colour': 'green'}

        # 1st run: E-STOP is pressed at step 200, RESET PROCESS at step 210 -> the cycle is aborted
        gui.update_status(process='RUNNING')
        gui._cycle_running = True
        ok = main.run_gui_cycle(env, arms, box, ink_pool, command, controllers=controllers)
        self.assertFalse(ok)
        self.assertEqual(gui.get_status_snapshot()['process'], 'READY')
        self.assertFalse(gui._cycle_running)
        self.assertFalse(gui._estopped())
        self.assertFalse(gui.abort_requested())

        # the main loop now sees the reset request: box back in storage, ink marks given back to the pool,
        # then finish_reset_process() sends the arms home                                              # CHANGED
        ink_pool.take()                                     # pretend the aborted cycle had laid a mark
        self.assertTrue(gui.consume_reset_request())
        main.reset_scene(box, ink_pool)
        gui.finish_reset_process()                                                                     # ADDED
        for arm in arms:                                                                               # MOVED here
            np.testing.assert_allclose(arm.robot.q, arm.HOME, atol=1e-9)
        np.testing.assert_allclose(box.pose.t[:2], STORAGE_XY, atol=1e-9)
        self.assertEqual(box.marks, [])
        self.assertEqual(ink_pool.used, 0)

        # 2nd run with the same box: runs to the end and uses the pool again
        env.estop_at = env.reset_at = None
        gui.update_status(process='RUNNING')
        gui._cycle_running = True
        ok = main.run_gui_cycle(env, arms, box, ink_pool, command, controllers=controllers)
        self.assertTrue(ok)
        self.assertEqual(gui.get_status_snapshot()['process'], 'COMPLETE')
        self.assertGreater(ink_pool.used, 0)

    def test_reset_process_after_a_completed_cycle_allows_another_run(self):
        env = FakeEnv()
        build_workcell(env)
        arms = make_arms()
        build_gui(env, arms)
        controllers = gui.get_controllers()
        box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        ink_pool = MarkPool(env, 400, lambda: Sphere(radius=0.0018, color=(0, 0, 0, 1)))
        command = {'mode': 'basic', 'shape': 'circle', 'colour': 'green'}

        gui.update_status(process='RUNNING')
        gui._cycle_running = True
        self.assertTrue(main.run_gui_cycle(env, arms, box, ink_pool, command, controllers=controllers))
        used_first = ink_pool.used
        self.assertGreater(used_first, 0)
        self.assertGreater(len(box.marks), 0)

        # RESET PROCESS after COMPLETE: box and marks are cleared, the second cycle lays the same number of marks
        self.assertTrue(gui.reset_process())
        self.assertTrue(gui.consume_reset_request())
        main.reset_scene(box, ink_pool)
        gui.finish_reset_process()                                                                     # ADDED
        self.assertEqual(ink_pool.used, 0)
        self.assertEqual(box.marks, [])
        np.testing.assert_allclose(box.pose.t[:2], STORAGE_XY, atol=1e-9)

        gui.update_status(process='RUNNING')
        gui._cycle_running = True
        self.assertTrue(main.run_gui_cycle(env, arms, box, ink_pool, command, controllers=controllers))
        self.assertEqual(ink_pool.used, used_first)

    def test_estop_without_reset_keeps_the_robots_held(self):
        env = FakeEnv()
        arms = make_arms()
        build_gui(env, arms)
        controllers = gui.get_controllers()
        gui._cycle_running = True
        gui.request_stop()
        arm = arms[0]
        before = arm.robot.q.copy()
        target = before.copy()
        target[0] += 0.05

        # play() would hold forever, so let the operator press RESET PROCESS after 5 held steps
        release = ScriptedEnv(estop_at=None, reset_at=5)
        with self.assertRaises(gui.CycleAborted):
            main.play(release, arm, [target], dt=0.02, controller=controllers[0])
        self.assertEqual(release.step_count, 5)

    def test_collision_latches_estop_and_reset_process_aborts_the_cycle(self):
        env = ScriptedEnv(estop_at=None, reset_at=5)
        arms = make_arms()
        build_gui(env, arms)
        controllers = gui.get_controllers()
        gui._cycle_running = True
        arm = arms[0]
        target = arm.HOME.copy()
        target[0] += 0.05

        # the arm would collide: play() must hold, the E-STOP is latched, RESET PROCESS (step 5) aborts
        with patch('gui.evaluate_safety', return_value=(UNSAFE_COLLISION, SAFE_SINGULARITY)):
            with self.assertRaises(gui.CycleAborted):
                main.play(env, arm, [target], dt=0.02, controller=controllers[0])
        self.assertEqual(env.step_count, 5)
        self.assertTrue(any('Safety stop' in m for m in gui.status['log']))
        self.assertFalse(gui._estopped())
        gui.finish_reset_process()                                                                     # ADDED
        np.testing.assert_allclose(arm.robot.q, arm.HOME, atol=1e-9)

class TestWristCameraModel(unittest.TestCase):
    def setUp(self):
        reset_gui_state()

    def test_make_camera_model_is_a_cuboid(self):
        self.assertIsInstance(workcell.make_camera_model(), Cuboid)

    def test_play_moves_the_camera_model_with_the_arm(self):
        arm = make_arms()[1]
        arm.cam_model = FakeMark()
        main.play(FakeEnv(), arm, [arm.HOME.copy()], dt=0.02)
        np.testing.assert_allclose(arm.cam_model.T, arm.cam_pose().A, atol=1e-9)

    def test_play_works_for_arms_without_a_camera_model(self):
        arm = make_arms()[0]
        main.play(FakeEnv(), arm, [arm.HOME.copy()], dt=0.02)


if __name__ == "__main__":
    unittest.main()