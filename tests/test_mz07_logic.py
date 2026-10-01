# Including libraries
import unittest
import numpy as np

from nachi_mz07_task import NachiShapeFill
from staubli_tx2_60_task import ShapeSpec

class TestNachiShapeFill(unittest.TestCase):

    def setUp(self):
        self.arm = NachiShapeFill()

    def test_home_within_limits(self):
        self.assertTrue(self.arm.within_limits(self.arm.HOME))

    def test_solve_ik_reaches_target_within_tolerance(self):
        x, y, z = 0.40, 0.0, 0.35
        q = self.arm.solve_ik(x, y, z)
        self.assertIsNotNone(q, "IK should find a solution for a reachable point")
        self.assertTrue(self.arm.within_limits(q))
        reached = self.arm.fk(q).t
        err_mm = 1000 * np.linalg.norm(reached - np.array([x, y, z]))
        self.assertLessEqual(err_mm, self.arm.IK_TOL_MM)

    def test_solve_ik_returns_none_for_unreachable_point(self):
        q = self.arm.solve_ik(10.0, 10.0, 10.0)
        self.assertIsNone(q)

    # -- RMRC descend --------------------------------------------------------

    def test_rmrc_descend_starts_at_q_start(self):
        q_start = self.arm.solve_ik(0.40, 0.0, 0.47)
        self.assertIsNotNone(q_start)
        traj = self.arm.rmrc_descend(q_start, 0.47, 0.36)
        np.testing.assert_allclose(traj[0], q_start, atol=1e-9)

    def test_rmrc_descend_reaches_target_height(self):
        z_start, z_end = 0.47, 0.36
        q_start = self.arm.solve_ik(0.40, 0.0, z_start)
        self.assertIsNotNone(q_start)
        traj = self.arm.rmrc_descend(q_start, z_start, z_end)
        final_z = self.arm.fk(traj[-1]).t[2]
        self.assertAlmostEqual(final_z, z_end, delta=0.003)

    def test_rmrc_descend_keeps_xy_roughly_fixed(self):
        z_start, z_end = 0.47, 0.36
        x, y = 0.40, 0.0
        q_start = self.arm.solve_ik(x, y, z_start)
        self.assertIsNotNone(q_start)
        traj = self.arm.rmrc_descend(q_start, z_start, z_end)
        for q in traj:
            pos = self.arm.fk(q).t
            self.assertAlmostEqual(pos[0], x, delta=0.005)
            self.assertAlmostEqual(pos[1], y, delta=0.005)

    def test_rmrc_descend_monotonic_in_z(self):
        z_start, z_end = 0.47, 0.36  # descending, z_start > z_end
        q_start = self.arm.solve_ik(0.40, 0.0, z_start)
        traj = self.arm.rmrc_descend(q_start, z_start, z_end)
        zs = [self.arm.fk(q).t[2] for q in traj]
        self.assertTrue(all(a >= b - 1e-6 for a, b in zip(zs, zs[1:])),
                        "z should not increase during a descend")

    # -- plan_stamp / check_stamp ---------------------------------------------

    def test_plan_stamp_has_three_segments(self):
        shape = ShapeSpec(cx=0.40, cy=0.0, size=0.02, kind='circle')
        plan = self.arm.plan_stamp(shape, z_top=0.35)
        labels = [label for label, _ in plan]
        self.assertEqual(labels, ['approach', 'descend', 'retract'])

    def test_plan_stamp_works_for_all_three_kinds(self):
        for kind in ['circle', 'square', 'triangle']:
            shape = ShapeSpec(cx=0.40, cy=0.0, size=0.02, kind=kind)
            plan = self.arm.plan_stamp(shape, z_top=0.35)
            labels = [label for label, _ in plan]
            self.assertEqual(labels, ['approach', 'descend', 'retract'])

    def test_descend_segment_reaches_standoff_height(self):
        shape = ShapeSpec(cx=0.40, cy=0.0, size=0.02, kind='circle')
        z_top = 0.35
        plan = self.arm.plan_stamp(shape, z_top)
        descend_traj = dict(plan)['descend']
        final_z = self.arm.fk(descend_traj[-1]).t[2]
        self.assertAlmostEqual(final_z, z_top + self.arm.STANDOFF, delta=0.003)

    def test_contact_point_lands_on_shape_centre(self):
        shape = ShapeSpec(cx=0.40, cy=0.0, size=0.02, kind='circle')
        z_top = 0.35
        plan = self.arm.plan_stamp(shape, z_top)
        descend_traj = dict(plan)['descend']
        point, on_lid = self.arm.lid_contact(z_top, q=descend_traj[-1])
        self.assertTrue(on_lid)
        err_mm = 1000 * np.hypot(point[0] - shape.cx, point[1] - shape.cy)
        self.assertLessEqual(err_mm, 5.0)

    def test_plan_home_returns_to_home_pose(self):
        self.arm.robot.q = np.deg2rad([30, -20, 40, 0, 30, 0])
        plan = self.arm.plan_home()
        final_q = plan[0][1][-1]
        np.testing.assert_allclose(final_q, self.arm.HOME, atol=1e-6)

    def test_check_stamp_true_for_reachable_shape(self):
        shape = ShapeSpec(cx=0.40, cy=0.0, size=0.02, kind='circle')
        self.assertTrue(self.arm.check_stamp(shape, z_top=0.35))

    def test_check_stamp_false_for_unreachable_shape(self):
        shape = ShapeSpec(cx=10.0, cy=10.0, size=0.02, kind='circle')
        self.assertFalse(self.arm.check_stamp(shape, z_top=10.0))

if __name__ == "__main__":
    unittest.main()