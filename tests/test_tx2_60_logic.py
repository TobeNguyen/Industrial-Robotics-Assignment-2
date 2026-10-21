import unittest
import numpy as np
from spatialmath import SE3
from staubli_tx2_60_task import StaubliShapeDraw, ShapeSpec
import sensing

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

    def _square_stroke(self, size, cx=0.35, cy=0.0):
        pts = np.array([[-size, -size], [size, -size], [size, size], [-size, size], [-size, -size]])
        return sensing.create_shape_model(pts, cx=cx, cy=cy, size=size)

    def test_plan_draw_strokes_labels_each_stroke(self):
        self.arm.ARC_POINTS = 60
        strokes = [self._square_stroke(0.02), self._square_stroke(0.01)]
        plan = self.arm.plan_draw_strokes(strokes, z_top=0.35)
        labels = [label for label, _ in plan]
        self.assertEqual(labels, [
            'stroke0_approach', 'stroke0_touch', 'stroke0_draw', 'stroke0_retract',
            'stroke1_approach', 'stroke1_touch', 'stroke1_draw', 'stroke1_retract'])

    def test_plan_draw_strokes_second_stroke_starts_where_first_ended(self):
        self.arm.ARC_POINTS = 60
        strokes = [self._square_stroke(0.02), self._square_stroke(0.01)]
        plan = self.arm.plan_draw_strokes(strokes, z_top=0.35)
        np.testing.assert_allclose(plan[4][1][0], plan[3][1][-1], atol=1e-9)

    def test_plan_draw_strokes_restores_robot_q(self):
        self.arm.ARC_POINTS = 60
        q_before = self.arm.robot.q.copy()
        self.arm.plan_draw_strokes([self._square_stroke(0.02)], z_top=0.35)
        np.testing.assert_allclose(self.arm.robot.q, q_before, atol=1e-12)

    def test_plan_draw_strokes_asks_for_a_correction_per_stroke(self):
        self.arm.ARC_POINTS = 60
        asked = []
        def correction_fn(i):
            asked.append(i)
            return None
        strokes = [self._square_stroke(0.02), self._square_stroke(0.01)]
        self.arm.plan_draw_strokes(strokes, z_top=0.35, correction_fn=correction_fn)
        self.assertEqual(asked, [0, 1])

    def test_plan_draw_strokes_correction_shifts_the_stroke(self):
        self.arm.ARC_POINTS = 60
        stroke = self._square_stroke(0.02)
        plain = self.arm.plan_draw_strokes([stroke], z_top=0.35)
        shifted = self.arm.plan_draw_strokes(
            [stroke], z_top=0.35, correction_fn=lambda i: SE3(0.01, 0.0, 0.0))
        p0 = self.arm.fk(dict(plain)['stroke0_draw'][0]).t
        p1 = self.arm.fk(dict(shifted)['stroke0_draw'][0]).t
        self.assertAlmostEqual(p1[0] - p0[0], 0.01, delta=0.0005)

    def test_check_strokes_true_for_reachable_strokes(self):
        self.arm.ARC_POINTS = 60
        strokes = [self._square_stroke(0.02), self._square_stroke(0.01)]
        self.assertTrue(self.arm.check_strokes(strokes, z_top=0.35))

    def test_check_strokes_false_for_unreachable_stroke(self):
        self.arm.ARC_POINTS = 60
        strokes = [self._square_stroke(0.02, cx=10.0, cy=10.0)]
        self.assertFalse(self.arm.check_strokes(strokes, z_top=10.0))

    def test_tool_to_cam_is_a_pure_translation(self):
        # plan_look() relies on the camera and the flange having the same orientation
        np.testing.assert_allclose(self.arm.TOOL_TO_CAM.R, np.eye(3), atol=1e-12)

    def test_cam_pose_is_flange_pose_times_tool_to_cam(self):
        q = self.arm.solve_ik(0.35, 0.0, 0.35)
        expected = self.arm.fk(q) * self.arm.TOOL_TO_CAM
        np.testing.assert_allclose(self.arm.cam_pose(q).A, expected.A, atol=1e-9)

    def test_plan_look_has_one_look_segment(self):
        plan = self.arm.plan_look(0.35, 0.0, 0.35, 0.15)
        self.assertEqual([label for label, _ in plan], ['look'])
        np.testing.assert_allclose(plan[0][1][0], self.arm.HOME, atol=1e-9)

    def test_plan_look_puts_the_camera_above_the_target(self):
        plan = self.arm.plan_look(0.35, 0.0, 0.35, 0.15)
        cam = self.arm.cam_pose(plan[0][1][-1])
        np.testing.assert_allclose(cam.t, [0.35, 0.0, 0.50], atol=0.0005)

    def test_plan_look_camera_looks_straight_down(self):
        plan = self.arm.plan_look(0.35, 0.0, 0.35, 0.15)
        cam = self.arm.cam_pose(plan[0][1][-1])
        np.testing.assert_allclose(cam.R[:, 2], [0.0, 0.0, -1.0], atol=0.001)

    def test_plan_look_raises_for_unreachable_target(self):
        with self.assertRaises(ValueError):
            self.arm.plan_look(10.0, 10.0, 10.0, 0.15)

    def test_check_look_true_and_false(self):
        self.assertTrue(self.arm.check_look(0.35, 0.0, 0.35, 0.15))
        self.assertFalse(self.arm.check_look(10.0, 10.0, 10.0, 0.15))


if __name__ == "__main__":
    unittest.main()