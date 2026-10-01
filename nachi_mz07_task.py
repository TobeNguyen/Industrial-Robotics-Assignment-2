# Including libraries
import os
import sys

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from math import pi

from ir_support import line_plane_intersection

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(REPO, 'robots', 'NachiMZ07'))

from mz07 import MZ07
from staubli_tx2_60_task import ShapeSpec

class NachiShapeFill:
    HOME = np.deg2rad([0, -40, 20, 0, 60, -180])

    APPROACH = 0.120
    STANDOFF = 0.010

    IK_ATTEMPTS = 8
    IK_TOL_MM = 0.05
    JOINT_WEIGHTS = np.array([3, 3, 2, 1, 1, 1])

    RMRC_STEP_MM = 1.0
    RMRC_DT = 0.02

    def __init__(self, base=None, q_home=None):
        self.robot = MZ07()
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

    # Flange pose in world coordinates for q (current q if None).
    def fk(self, q=None):
        return self.robot.fkine(self.robot.q if q is None else q)

    # Checks whether q sits within this robot's joint limits.
    def within_limits(self, q):
        q = np.asarray(q)
        return bool(np.all(q >= self.robot.qlim[0] - 1e-9)
                    and np.all(q <= self.robot.qlim[1] + 1e-9))

    # Solves for joint angles at world (x, y, z), tool-down, validating each candidate against FK and joint limits before keeping the closest to the seed (stops the arm flipping to a different configuration between approach and retract).
    def solve_ik(self, x, y, z, q_seed=None, attempts=None):
        goal = SE3(x, y, z) * SE3.Ry(pi)
        target = np.array([x, y, z], dtype=float)
        seed = self.robot.q if q_seed is None else np.asarray(q_seed)

        best, best_cost = None, np.inf
        for _ in range(self.IK_ATTEMPTS if attempts is None else attempts):
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

    # Quintic joint-space trajectory, step count scaled to the largest joint travel so every move keeps roughly the same speed. Used for approach/retract, where only the endpoint pose matters.
    def joint_traj(self, q_goal, steps=None, q_start=None):
        start = self.robot.q if q_start is None else np.asarray(q_start)
        goal = np.asarray(q_goal)
        if steps is None:
            span = np.degrees(np.max(np.abs(goal - start)))
            steps = int(np.clip(round(span / self.MAX_STEP_DEG), 20, 400))
        return rtb.jtraj(start, goal, steps).q

    # Resolved-motion rate control: descends straight down (fixed x, y, orientation) from z_start to z_end by turning a desired Cartesian velocity into joint velocity via the Jacobian pseudo-inverse and integrating it, rather than solving IK at each waypoint. This is the RMRC segment of the stamp motion - it is what keeps the tool travelling in a straight vertical line on the way to contact.
    def rmrc_descend(self, q_start, z_start, z_end, steps=None, dt=None):
        dt = self.RMRC_DT if dt is None else dt
        if steps is None:
            steps = int(np.clip(round(1000 * abs(z_end - z_start) / self.RMRC_STEP_MM), 10, 200))
        dz = (z_end - z_start) / steps

        q = np.asarray(q_start, dtype=float).copy()
        traj = [q.copy()]
        v = np.array([0.0, 0.0, dz / dt, 0.0, 0.0, 0.0])
        for _ in range(steps):
            J = self.robot.jacob0(q)
            qdot = np.linalg.pinv(J) @ v
            q = q + qdot * dt
            traj.append(q.copy())
        return np.vstack(traj)

    # Where the tool axis crosses the lid, by line-plane intersection - this is where the stamp actually lands, not the commanded flange target.
    def lid_contact(self, lid_z, q=None, half_width=None):
        T = self.fk(q)
        approach = T.a
        p1 = T.t - approach * 0.10
        p2 = T.t + approach * 0.10
        point, check = line_plane_intersection(np.array([0.0, 0.0, 1.0]),
                                            np.array([0.0, 0.0, lid_z]), p1, p2)
        if check != 1:
            return point, False
        if half_width is not None:
            centre = getattr(self, '_lid_centre', (0.0, 0.0))
            if (abs(point[0] - centre[0]) > half_width
                    or abs(point[1] - centre[1]) > half_width):
                return point, False
        return point, True

    # Lines up over the shape's centre, descends onto it with RMRC, then retracts. The caller (main.py) is expected to drop a single mark at the contact point once the 'descend' segment has played - this method only plans the motion, it does not place any mark itself.
    def plan_stamp(self, shape, z_top, steps=None):
        self._lid_centre = (shape.cx, shape.cy)
        z_contact = z_top + self.STANDOFF
        cursor = self.robot.q.copy()
        plan = []

        q = self.solve_ik(shape.cx, shape.cy, z_top + self.APPROACH, q_seed=cursor)
        if q is None:
            raise ValueError('NachiShapeFill: cannot reach the approach pose')
        plan.append(('approach', self.joint_traj(q, steps, q_start=cursor)))
        cursor = q

        descend = self.rmrc_descend(cursor, z_top + self.APPROACH, z_contact, steps=steps)
        plan.append(('descend', descend))
        cursor = descend[-1]

        q = self.solve_ik(shape.cx, shape.cy, z_top + self.APPROACH, q_seed=cursor)
        if q is None:
            raise ValueError('NachiShapeFill: cannot reach the retract pose')
        plan.append(('retract', self.joint_traj(q, steps, q_start=cursor)))
        return plan

    # Folds the arm back to its home pose.
    def plan_home(self, steps=None):
        return [('home', self.joint_traj(self.HOME, steps))]

    # Confirms the stamp plan is reachable and that the point of contact at the end of 'descend' sits on the shape's own centre, within a few millimetres.
    def check_stamp(self, shape, z_top, tol_mm=5.0):
        try:
            plan = self.plan_stamp(shape, z_top)
        except ValueError as exc:
            print(f'  {exc}')
            return False
        descend_traj = dict(plan)['descend']
        point, on_lid = self.lid_contact(z_top, q=descend_traj[-1])
        err_mm = 1000 * np.hypot(point[0] - shape.cx, point[1] - shape.cy)
        ok = on_lid and err_mm <= tol_mm
        print(f'  NachiShapeFill: {shape.kind} stamped, {len(descend_traj)} descend steps, '
            f'contact {err_mm:.1f} mm off centre' + ('' if ok else ' — FAILED'))
        return ok
    
if __name__ == "__main__":
    arm = NachiShapeFill()
    shape = ShapeSpec(cx=0.40, cy=0.0, size=0.02)
    print(arm.robot)
    print('flange at home:', np.round(arm.fk().t, 4))
    arm.check_stamp(shape, z_top=0.35)