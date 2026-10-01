import unittest
import numpy as np

from staubli_tx2_60_task import StaubliShapeDraw, ShapeSpec


class TestShapeSpec(unittest.TestCase):

    def test_circle_boundary_points_stay_on_radius(self):
        shape = ShapeSpec(cx=0.35, cy=0.0, size=0.035, kind='circle')
        pts = shape.boundary_points(n=36)
        for x, y in pts:
            r = np.hypot(x - shape.cx, y - shape.cy)
            self.assertAlmostEqual(r, shape.size, delta=1e-9)

    def test_circle_boundary_is_closed_loop(self):
        shape = ShapeSpec(cx=0.35, cy=0.0, size=0.035, kind='circle')
        pts = shape.boundary_points(n=36)
        np.testing.assert_allclose(pts[0], pts[-1], atol=1e-9)

    def test_square_boundary_points_lie_within_bounding_box(self):
        shape = ShapeSpec(cx=0.4, cy=0.1, size=0.02, kind='square')
        pts = shape.boundary_points(n=40)
        for x, y in pts:
            self.assertLessEqual(abs(x - shape.cx), shape.size + 1e-9)
            self.assertLessEqual(abs(y - shape.cy), shape.size + 1e-9)

    def test_square_boundary_is_closed_loop(self):
        shape = ShapeSpec(cx=0.4, cy=0.1, size=0.02, kind='square')
        pts = shape.boundary_points(n=40)
        np.testing.assert_allclose(pts[0], pts[-1], atol=1e-9)

    def test_triangle_boundary_is_closed_loop(self):
        shape = ShapeSpec(cx=0.3, cy=0.0, size=0.025, kind='triangle')
        pts = shape.boundary_points(n=30)
        np.testing.assert_allclose(pts[0], pts[-1], atol=1e-9)

    def test_half_width_at_matches_circle_chord(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.03, kind='circle')
        dy = 0.01
        expected = np.sqrt(shape.size ** 2 - dy ** 2)
        self.assertAlmostEqual(shape.half_width_at(dy), expected, places=9)

    def test_half_width_at_zero_outside_circle_extent(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.03, kind='circle')
        self.assertEqual(shape.half_width_at(0.05), 0.0)

    def test_half_width_at_constant_for_square(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='square')
        self.assertAlmostEqual(shape.half_width_at(0.0), shape.size)
        self.assertAlmostEqual(shape.half_width_at(0.019), shape.size)
        self.assertEqual(shape.half_width_at(0.021), 0.0)

    def test_half_width_at_shrinks_to_zero_at_triangle_apex(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='triangle')
        self.assertAlmostEqual(shape.half_width_at(shape.size), 0.0, places=9)
        self.assertAlmostEqual(shape.half_width_at(-shape.size), shape.size, places=9)

    def test_unknown_kind_raises(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02, kind='hexagon')
        with self.assertRaises(ValueError):
            shape.boundary_points()
        with self.assertRaises(ValueError):
            shape.half_width_at(0.0)

    def test_random_kind_is_one_of_three(self):
        shape = ShapeSpec(cx=0.0, cy=0.0, size=0.02)
        self.assertIn(shape.kind, ['circle', 'square', 'triangle'])


class TestStaubliShapeDraw(unittest.TestCase):

    def setUp(self):
        self.arm = StaubliShapeDraw()

    def test_home_within_limits(self):
        self.assertTrue(self.arm.within_limits(self.arm.HOME))

    def test_solve_ik_reaches_target_within_tolerance(self):
        x, y, z = 0.35, 0.0, 0.35
        q = self.arm.solve_ik(x, y, z)
        self.assertIsNotNone(q, "IK should find a solution for a reachable point")
        self.assertTrue(self.arm.within_limits(q))
        reached = self.arm.fk(q).t
        err_mm = 1000 * np.linalg.norm(reached - np.array([x, y, z]))
        self.assertLessEqual(err_mm, self.arm.IK_TOL_MM)

    def test_solve_ik_returns_none_for_unreachable_point(self):
        q = self.arm.solve_ik(10.0, 10.0, 10.0)
        self.assertIsNone(q)

    def test_plan_draw_shape_circle_has_four_segments(self):
        shape = ShapeSpec(cx=0.35, cy=0.0, size=0.035, kind='circle')
        plan = self.arm.plan_draw_shape(shape, z_top=0.35)
        labels = [label for label, _ in plan]
        self.assertEqual(labels, ['approach', 'touch', 'draw', 'retract'])

    def test_plan_draw_shape_square_has_four_segments(self):
        shape = ShapeSpec(cx=0.35, cy=0.0, size=0.02, kind='square')
        plan = self.arm.plan_draw_shape(shape, z_top=0.35)
        labels = [label for label, _ in plan]
        self.assertEqual(labels, ['approach', 'touch', 'draw', 'retract'])

    def test_plan_draw_shape_triangle_has_four_segments(self):
        shape = ShapeSpec(cx=0.35, cy=0.0, size=0.02, kind='triangle')
        plan = self.arm.plan_draw_shape(shape, z_top=0.35)
        labels = [label for label, _ in plan]
        self.assertEqual(labels, ['approach', 'touch', 'draw', 'retract'])

    def test_draw_segment_reaches_z_top(self):
        shape = ShapeSpec(cx=0.35, cy=0.0, size=0.02, kind='circle')
        z_top = 0.35
        plan = self.arm.plan_draw_shape(shape, z_top)
        draw_traj = dict(plan)['draw']
        final_pos = self.arm.fk(draw_traj[-1]).t
        self.assertAlmostEqual(final_pos[2], z_top, delta=0.001)

    def test_plan_home_returns_to_home_pose(self):
        self.arm.robot.q = np.deg2rad([30, -20, 40, 0, 30, 0])
        plan = self.arm.plan_home()
        final_q = plan[0][1][-1]
        np.testing.assert_allclose(final_q, self.arm.HOME, atol=1e-6)

    def test_check_shape_true_for_reachable_shape(self):
        shape = ShapeSpec(cx=0.35, cy=0.0, size=0.02, kind='circle')
        self.assertTrue(self.arm.check_shape(shape, z_top=0.35))

    def test_check_shape_false_for_unreachable_shape(self):
        shape = ShapeSpec(cx=10.0, cy=10.0, size=0.02, kind='circle')
        self.assertFalse(self.arm.check_shape(shape, z_top=10.0))


if __name__ == "__main__":
    unittest.main()