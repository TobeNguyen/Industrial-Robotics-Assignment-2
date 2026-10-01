# test_rs007n_logic.py
import unittest
import numpy as np

from kawasaki_rs007n_task import KawasakiPickPlace


class TestKawasakiPickPlace(unittest.TestCase):

    def setUp(self):
        self.arm = KawasakiPickPlace()

    def test_home_within_limits(self):
        self.assertTrue(self.arm.within_limits(self.arm.HOME))

    def test_fk_matches_home_position(self):
        # Sanity check: FK at q=HOME should return a finite 3D point.
        pos = self.arm.fk().t
        self.assertEqual(pos.shape, (3,))
        self.assertTrue(np.all(np.isfinite(pos)))

    def test_solve_ik_reaches_target_within_tolerance(self):
        x, y, z = 0.45, 0.0, 0.25
        q = self.arm.solve_ik(x, y, z)
        self.assertIsNotNone(q, "IK should find a solution for a reachable point")
        self.assertTrue(self.arm.within_limits(q))
        reached = self.arm.fk(q).t
        err_mm = 1000 * np.linalg.norm(reached - np.array([x, y, z]))
        self.assertLessEqual(err_mm, self.arm.IK_TOL_MM)

    def test_solve_ik_returns_none_for_unreachable_point(self):
        # Far outside any real RS007N workspace.
        q = self.arm.solve_ik(10.0, 10.0, 10.0)
        self.assertIsNone(q)

    def test_joint_traj_starts_and_ends_correctly(self):
        q_goal = np.deg2rad([10, 10, 10, 10, 10, 10])
        traj = self.arm.joint_traj(q_goal, q_start=self.arm.HOME)
        np.testing.assert_allclose(traj[0], self.arm.HOME, atol=1e-9)
        np.testing.assert_allclose(traj[-1], q_goal, atol=1e-6)

    def test_joint_traj_step_count_scales_with_distance(self):
        near_goal = self.arm.HOME + np.deg2rad([1, 0, 0, 0, 0, 0])
        far_goal = self.arm.HOME + np.deg2rad([90, 0, 0, 0, 0, 0])
        near_traj = self.arm.joint_traj(near_goal, q_start=self.arm.HOME)
        far_traj = self.arm.joint_traj(far_goal, q_start=self.arm.HOME)
        self.assertLess(near_traj.shape[0], far_traj.shape[0])

    def test_plan_pick_has_three_segments_and_reaches_target_xy(self):
        x, y, z_top = 0.45, 0.0, 0.25
        plan = self.arm.plan_pick(x, y, z_top)
        labels = [label for label, _ in plan]
        self.assertEqual(labels, ['approach', 'descend', 'lift'])

        # 'descend' should land at the part's actual (x, y, z_top).
        descend_traj = dict(plan)['descend']
        final_pos = self.arm.fk(descend_traj[-1]).t
        self.assertAlmostEqual(final_pos[2], z_top, delta=0.001)

    def test_plan_place_has_three_segments(self):
        plan = self.arm.plan_place(0.40, 0.10, 0.25)
        labels = [label for label, _ in plan]
        self.assertEqual(labels, ['traverse', 'descend', 'retract'])

    def test_plan_home_returns_to_home_pose(self):
        # Move away first, then plan home and check the trajectory ends at HOME.
        self.arm.robot.q = np.deg2rad([30, -20, 40, 0, 30, 0])
        plan = self.arm.plan_home()
        final_q = plan[0][1][-1]
        np.testing.assert_allclose(final_q, self.arm.HOME, atol=1e-6)

    def test_check_stations_true_for_reachable_points(self):
        stations = [
            ('pick', 0.45, 0.0, 0.25),
            ('place', 0.40, 0.10, 0.25),
        ]
        self.assertTrue(self.arm.check_stations(stations))

    def test_check_stations_false_for_unreachable_point(self):
        stations = [('impossible', 10.0, 10.0, 10.0)]
        self.assertFalse(self.arm.check_stations(stations))


if __name__ == "__main__":
    unittest.main()