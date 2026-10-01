# Including libraries
import unittest
import numpy as np
from spatialmath import SE3
from spatialgeometry import Cylinder, Cuboid, Mesh, Sphere

from workcell import (
    BOX, BOX_TOP, STORAGE_XY, FINISH_XY, STAND_TOP,
    Y_LOAD, Y_MARK, Y_PAINT, SHAPE_SIZE, PAINT_MARGIN,
    MarkedBox, MarkPool, make_arms, build_workcell, make_stamp_shape,
)
from staubli_tx2_60_task import ShapeSpec
import main

class FakeEnv:
    def __init__(self):
        self.added = []
        self.step_count = 0

    def add(self, shape):
        self.added.append(shape)
        return shape

    def step(self, dt=0.02):
        self.step_count += 1

    def set_camera_pose(self, *args, **kwargs):
        pass

    def hold(self):
        pass

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

    def test_triangle_and_square_stamps_are_different_types(self):
        square = make_stamp_shape(ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='square'))
        triangle = make_stamp_shape(ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='triangle'))
        self.assertNotEqual(type(square), type(triangle))

    def test_larger_shape_produces_larger_stamp(self):
        small = make_stamp_shape(ShapeSpec(cx=0.0, cy=0.0, size=0.01, kind='circle'))
        big = make_stamp_shape(ShapeSpec(cx=0.0, cy=0.0, size=0.03, kind='circle'))
        self.assertGreater(big.radius, small.radius)

    def test_unknown_kind_raises(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='hexagon')
        with self.assertRaises(ValueError):
            make_stamp_shape(shape)

class TestConvey(unittest.TestCase):
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
        self.assertEqual(len(self.box.marks), 0)

    def test_respects_every_parameter(self):
        arm = FakeArm(contact_point=(0.0, Y_MARK, BOX_TOP), on_lid=True)
        pool = FakePool()
        on_frame = main.make_marker(self.env, self.box, pool, every=3, lid_z=BOX_TOP, z_lift=0.001)
        for k in range(9):
            on_frame(arm, k)
        self.assertEqual(pool.taken, 3)

    def test_max_radius_rejects_points_outside_shape(self):
        centre = (0.0, Y_MARK)
        arm = FakeArm(contact_point=(0.05, Y_MARK, BOX_TOP), on_lid=True)
        pool = FakePool()
        on_frame = main.make_marker(self.env, self.box, pool, every=1, lid_z=BOX_TOP,
                                    z_lift=0.001, max_radius=0.03, centre=centre)
        on_frame(arm, k=0)
        self.assertEqual(pool.taken, 0)

    def test_max_radius_accepts_points_inside_shape(self):
        centre = (0.0, Y_MARK)
        arm = FakeArm(contact_point=(0.01, Y_MARK, BOX_TOP), on_lid=True)
        pool = FakePool()
        on_frame = main.make_marker(self.env, self.box, pool, every=1, lid_z=BOX_TOP,
                                    z_lift=0.001, max_radius=0.03, centre=centre)
        on_frame(arm, k=0)
        self.assertEqual(pool.taken, 1)

    def test_pool_exhaustion_adds_no_mark(self):
        class EmptyPool:
            def take(self):
                return None

        arm = FakeArm(contact_point=(0.0, Y_MARK, BOX_TOP), on_lid=True)
        on_frame = main.make_marker(self.env, self.box, EmptyPool(), every=1,
                                    lid_z=BOX_TOP, z_lift=0.001)
        on_frame(arm, k=0)
        self.assertEqual(len(self.box.marks), 0)

class TestRunStamp(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        _, _, self.arm3 = make_arms()
        self.box = MarkedBox(self.env, 0.0, Y_PAINT, STAND_TOP)

    def test_run_stamp_places_exactly_one_mark(self):
        shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=0.02, kind='circle')
        stamped = main.run_stamp(self.env, self.arm3, self.box, shape, BOX_TOP)
        self.assertTrue(stamped)
        self.assertEqual(len(self.box.marks), 1)

    def test_stamp_mark_size_matches_shape(self):
        shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=0.02, kind='circle')
        main.run_stamp(self.env, self.arm3, self.box, shape, BOX_TOP)
        placed_shape, _ = self.box.marks[0]
        self.assertAlmostEqual(placed_shape.radius, shape.size, places=6)

    def test_run_stamp_places_a_mesh_for_triangle(self):
        shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=0.02, kind='triangle')
        main.run_stamp(self.env, self.arm3, self.box, shape, BOX_TOP)
        placed_shape, _ = self.box.marks[0]
        self.assertIsInstance(placed_shape, Mesh)
        self.assertNotIsInstance(placed_shape, Cuboid)

    def test_run_stamp_works_for_all_three_kinds(self):
        for kind in ['circle', 'square', 'triangle']:
            env = FakeEnv()
            _, _, arm3 = make_arms()
            box = MarkedBox(env, 0.0, Y_PAINT, STAND_TOP)
            shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=0.02, kind=kind)
            stamped = main.run_stamp(env, arm3, box, shape, BOX_TOP)
            self.assertTrue(stamped, f'stamp failed for kind={kind}')
            self.assertEqual(len(box.marks), 1)

    def test_run_stamp_false_for_unreachable_shape(self):
        shape = ShapeSpec(cx=10.0, cy=10.0, size=0.02, kind='circle')
        with self.assertRaises(ValueError):
            main.run_stamp(self.env, self.arm3, self.box, shape, 10.0)

class TestRunPickAndPlace(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.arm1, self.arm2, self.arm3 = make_arms()
        self.box = MarkedBox(self.env, *STORAGE_XY, STAND_TOP)

    def test_run_pick_returns_a_grasp_transform(self):
        grasp = main.run_pick(self.env, self.arm1, self.box, *STORAGE_XY, STAND_TOP + BOX[2])
        self.assertIsInstance(grasp, SE3)

    def test_box_follows_flange_after_pick(self):
        grasp = main.run_pick(self.env, self.arm1, self.box, *STORAGE_XY, STAND_TOP + BOX[2])
        expected = self.arm1.fk() * grasp
        np.testing.assert_allclose(self.box.pose.t, expected.t, atol=1e-6)

    def test_run_place_releases_grasp(self):
        grasp = main.run_pick(self.env, self.arm1, self.box, *STORAGE_XY, STAND_TOP + BOX[2])
        grasp = main.run_place(self.env, self.arm1, self.box, grasp, 0.0, Y_LOAD, BOX_TOP)
        self.assertIsNone(grasp)

    def test_box_ends_at_place_target(self):
        grasp = main.run_pick(self.env, self.arm1, self.box, *STORAGE_XY, STAND_TOP + BOX[2])
        main.run_place(self.env, self.arm1, self.box, grasp, 0.0, Y_LOAD, BOX_TOP)
        self.assertAlmostEqual(self.box.pose.t[0], 0.0, delta=0.001)
        self.assertAlmostEqual(self.box.pose.t[1], Y_LOAD, delta=0.001)
        self.assertAlmostEqual(self.box.pose.t[2], BOX_TOP - BOX[2] / 2, delta=0.001)

class TestRunPlanWithMarking(unittest.TestCase):
    def test_marking_hook_only_fires_during_named_segment(self):
        env = FakeEnv()
        arm1, arm2, arm3 = make_arms()
        calls = {'draw': 0}

        def hook(arm, k):
            calls['draw'] += 1

        shape = ShapeSpec(cx=0.0, cy=Y_MARK, size=SHAPE_SIZE, kind='circle')
        plan = arm2.plan_draw_shape(shape, BOX_TOP)
        main.run_plan(env, arm2, plan, dt=0.001, marking=('draw', hook))

        draw_len = dict(plan)['draw'].shape[0]
        self.assertEqual(calls['draw'], draw_len)

class TestRunCycle(unittest.TestCase):
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
        self.assertGreater(ink_pool.used, 0, "draw station should have laid at least one ink mark")
        self.assertEqual(len(box.marks) - ink_pool.used, 1,
                        "paint station should have placed exactly one stamp mark")

    def test_box_ends_near_finish_stand(self):
        env, arms, box, ink_pool = self._fresh()
        main.run_cycle(env, arms, box, ink_pool)
        self.assertAlmostEqual(box.pose.t[0], FINISH_XY[0], delta=0.01)
        self.assertAlmostEqual(box.pose.t[1], FINISH_XY[1], delta=0.01)

    def test_repeated_cycles_keep_using_the_shared_ink_pool(self):
        env, arms, box1, ink_pool = self._fresh()
        main.run_cycle(env, arms, box1, ink_pool)
        used_ink_after_first = ink_pool.used

        box2 = MarkedBox(env, *STORAGE_XY, STAND_TOP)
        main.run_cycle(env, arms, box2, ink_pool)
        self.assertGreaterEqual(ink_pool.used, used_ink_after_first)

class TestCheckLayout(unittest.TestCase):
    def test_check_layout_passes_for_default_stations(self):
        self.assertTrue(main.check_layout())


if __name__ == "__main__":
    unittest.main()