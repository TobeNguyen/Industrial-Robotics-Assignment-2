# Including libraries
import os
import sys

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from math import pi

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(REPO, 'robots', 'KawasakiRS007N'))

from rs007 import RS007N

class KawasakiPickPlace:
    HOME = np.deg2rad([0, 20, 70, 0, 90, 0])

    APPROACH = 0.120
    IK_ATTEMPTS = 8
    IK_TOL_MM = 0.05

    JOINT_WEIGHTS = np.array([3, 3, 2, 1, 1, 1])

    def __init__(self, base=None, q_home=None):
        self.robot = RS007N()
        if base is not None:
            self.robot.base = base
        if q_home is not None:
            self.HOME = np.asarray(q_home, dtype=float)
        self.robot.q = self.HOME.copy()

    # Current joint vector, exposed for convenience.
    @property
    def q(self):
        return self.robot.q

    # Adds the robot mesh to a Swift environment.
    def add_to_env(self, env):
        self.robot.add_to_env(env)

    # Flange pose in world coordinates for joint angles q (current q if None).
    def fk(self, q=None):
        return self.robot.fkine(self.robot.q if q is None else q)

    # Checks whether q sits within this robot's joint limits.
    def within_limits(self, q):
        q = np.asarray(q)
        return bool(np.all(q >= self.robot.qlim[0] - 1e-9)
                    and np.all(q <= self.robot.qlim[1] + 1e-9))

    # Solves for joint angles that place the flange at world (x, y, z), tool-down, validating each candidate against FK and joint limits before keeping the one closest to the seed.
    def solve_ik(self, x, y, z, q_seed=None):
        goal = SE3(x, y, z) * SE3.Ry(pi)
        target = np.array([x, y, z], dtype=float)
        seed = self.robot.q if q_seed is None else np.asarray(q_seed)

        best, best_cost = None, np.inf
        for _ in range(self.IK_ATTEMPTS):
            sol = self.robot.ikine_LM(goal, q0=seed, joint_limits=True, tol=1e-10)
            if not sol.success:
                continue
            if 1000 * np.linalg.norm(self.robot.fkine(sol.q).t - target) > self.IK_TOL_MM:
                continue
            if not self.within_limits(sol.q):
                continue
            cost = float(np.sum(self.JOINT_WEIGHTS * np.abs(sol.q - seed)))
            if cost < best_cost:
                best, best_cost = sol.q, cost
        return best

    MAX_STEP_DEG = 1.5

    # Builds a quintic joint-space trajectory from q_start (or current q) to q_goal, sizing the step count to the largest joint travel so every move keeps roughly the same speed.
    def joint_traj(self, q_goal, steps=None, q_start=None):
        start = self.robot.q if q_start is None else np.asarray(q_start)
        goal = np.asarray(q_goal)
        if steps is None:
            span = np.degrees(np.max(np.abs(goal - start)))
            steps = int(np.clip(round(span / self.MAX_STEP_DEG), 20, 400))
        return rtb.jtraj(start, goal, steps).q

    # One IK solve plus the trajectory that reaches it, returning (traj, q_end).
    def _segment_to(self, x, y, z, steps, cursor):
        q = self.solve_ik(x, y, z, q_seed=cursor)
        if q is None:
            raise ValueError(f'{type(self).__name__}: cannot reach ({x:.3f}, {y:.3f}, {z:.3f})')
        return self.joint_traj(q, steps, q_start=cursor), q

    # Lines up above the part, descends onto it, and lifts clear; caller attaches the object to the flange once 'descend' has played.
    def plan_pick(self, x, y, z_top, steps=None):
        cursor = self.robot.q.copy()
        plan = []
        for label, height in [('approach', z_top + self.APPROACH),
                            ('descend', z_top),
                            ('lift', z_top + self.APPROACH)]:
            traj, cursor = self._segment_to(x, y, height, steps, cursor)
            plan.append((label, traj))
        return plan

    # Carries the part over the drop point, lowers it, and retracts; caller releases the object once 'descend' has played.
    def plan_place(self, x, y, z_top, steps=None):
        cursor = self.robot.q.copy()
        plan = []
        for label, height in [('traverse', z_top + self.APPROACH),
                            ('descend', z_top),
                            ('retract', z_top + self.APPROACH)]:
            traj, cursor = self._segment_to(x, y, height, steps, cursor)
            plan.append((label, traj))
        return plan

    # Folds the arm back to its home pose.
    def plan_home(self, steps=None):
        return [('home', self.joint_traj(self.HOME, steps))]

    # Confirms every (name, x, y, z_top) station is still reachable from home; run after moving the robot or furniture.
    def check_stations(self, stations):
        ok = True
        cursor = self.HOME.copy()
        for name, x, y, z_top in stations:
            for height in (z_top + self.APPROACH, z_top):
                q = self.solve_ik(x, y, height, q_seed=cursor)
                if q is None:
                    print(f'  {name:16s} z={height:.3f}  UNREACHABLE')
                    ok = False
                else:
                    cursor = q
        if ok:
            print(f'  {type(self).__name__}: all {len(stations)} stations reachable')
        return ok
    
if __name__ == "__main__":
    arm = KawasakiPickPlace()
    print(arm.robot)
    print('flange at home:', np.round(arm.fk().t, 4))
    plan = arm.plan_pick(0.45, 0.0, 0.25)
    for label, traj in plan:
        print(f'  {label:10s} {traj.shape[0]} steps -> {np.round(arm.fk(traj[-1]).t, 4)}')