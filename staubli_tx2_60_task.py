# Including libraries
import os
import sys
import random

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from math import pi

from ir_support import line_plane_intersection

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(REPO, 'robots', 'StaubliTX2_60'))

from tx2_60 import TX2_60

# Single shared description of the shape to draw and paint; StaubliShapeDraw and NachiShapeFill both read the same ShapeSpec instance so the traced boundary and the filled region can never disagree, and a future shape only needs to add boundary_points/half_width_at to work with both robots unchanged.
class ShapeSpec:

    def __init__(self, cx, cy, size, kind=None):
        self.kind = kind or random.choice(['circle', 'square', 'triangle'])
        self.cx = cx
        self.cy = cy
        self.size = size

    def _corners(self):
        s = self.size
        if self.kind == 'square':
            return [(-s, -s), (s, -s), (s, s), (-s, s), (-s, -s)]
        if self.kind == 'triangle':
            return [(0, s), (-s, -s), (s, -s), (0, s)]
        raise ValueError(f'_corners has no polygon for kind={self.kind!r}')

    def boundary_points(self, n=360):
        if self.kind == 'circle':
            t = np.linspace(0, 2 * np.pi, n)
            return [(self.cx + self.size * np.cos(a),
                     self.cy + self.size * np.sin(a)) for a in t]

        corners = self._corners()
        segments = len(corners) - 1
        pts_per_seg = max(2, n // segments)
        pts = []
        for i in range(segments):
            x0, y0 = corners[i]
            x1, y1 = corners[i + 1]
            last_segment = (i == segments - 1)
            for t in np.linspace(0, 1, pts_per_seg, endpoint=last_segment):
                pts.append((self.cx + x0 + (x1 - x0) * t,
                            self.cy + y0 + (y1 - y0) * t))
        return pts

    def half_width_at(self, dy):
        s = self.size
        if self.kind == 'circle':
            return np.sqrt(max(s ** 2 - dy ** 2, 0.0))
        if self.kind == 'square':
            return s if abs(dy) <= s else 0.0
        if self.kind == 'triangle':
            if dy < -s or dy > s:
                return 0.0
            return s * (s - dy) / (2 * s)
        raise ValueError(f'half_width_at has no rule for kind={self.kind!r}')
    
class StaubliShapeDraw:
    HOME = np.deg2rad([0, 30, 70, -180, -80, 0])

    APPROACH = 0.120
    ARC_POINTS = 360
    BLEND = 0
    PATH_ATTEMPTS = 3

    IK_ATTEMPTS = 8
    IK_TOL_MM = 0.05
    JOINT_WEIGHTS = np.array([3, 3, 2, 1, 1, 1])

    def __init__(self, base=None, q_home=None):
        self.robot = TX2_60()
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

    # Solves for joint angles at world (x, y, z), tool-down, validating each candidate against FK and joint limits before keeping the one closest to the seed (keeps the whole shape on one arm configuration).
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

    # Quintic joint-space trajectory, step count scaled to the largest joint travel so every move keeps roughly the same speed.
    def joint_traj(self, q_goal, steps=None, q_start=None):
        start = self.robot.q if q_start is None else np.asarray(q_start)
        goal = np.asarray(q_goal)
        if steps is None:
            span = np.degrees(np.max(np.abs(goal - start)))
            steps = int(np.clip(round(span / self.MAX_STEP_DEG), 20, 400))
        return rtb.jtraj(start, goal, steps).q

    # Follows a list of world (x, y, z) points tool-down, each solved from the previous answer so the arm stays on one IK branch.
    def cartesian_path(self, points, q_start=None, blend=None):
        blend = self.BLEND if blend is None else blend
        cursor = (self.robot.q if q_start is None else np.asarray(q_start)).copy()
        out = [cursor.reshape(1, -1)]
        for x, y, z in points:
            q = self.solve_ik(x, y, z, q_seed=cursor, attempts=self.PATH_ATTEMPTS)
            if q is None:
                raise ValueError(f'StaubliShapeDraw: cannot reach ({x:.3f}, {y:.3f}, {z:.3f})')
            out.append(q.reshape(1, -1) if blend == 0 else rtb.jtraj(cursor, q, blend + 1).q[1:])
            cursor = q
        return np.vstack(out)

    # Where the pen axis crosses the lid, by line-plane intersection — the contact point actually reached, residual IK error included.
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

    # Lines up over the lid, touches down at the shape's start point, traces shape.boundary_points(), then lifts clear — works for any ShapeSpec kind unchanged.
    def plan_draw_shape(self, shape, z_top, steps=None):
        self._lid_centre = (shape.cx, shape.cy)
        pts_xy = shape.boundary_points(self.ARC_POINTS)
        pts = [(x, y, z_top) for x, y in pts_xy]
        start_x, start_y, _ = pts[0]
        cursor = self.robot.q.copy()
        plan = []

        for label, (x, y, z) in [('approach', (start_x, start_y, z_top + self.APPROACH)),
                                ('touch', (start_x, start_y, z_top))]:
            q = self.solve_ik(x, y, z, q_seed=cursor)
            if q is None:
                raise ValueError(f'StaubliShapeDraw: cannot reach the {label} pose')
            plan.append((label, self.joint_traj(q, steps, q_start=cursor)))
            cursor = q

        plan.append(('draw', self.cartesian_path(pts, q_start=cursor)))
        cursor = plan[-1][1][-1]

        q = self.solve_ik(start_x, start_y, z_top + self.APPROACH, q_seed=cursor)
        plan.append(('retract', self.joint_traj(q, steps, q_start=cursor)))
        return plan

    # Folds the arm back to its home pose.
    def plan_home(self, steps=None):
        return [('home', self.joint_traj(self.HOME, steps))]

    # Confirms the whole shape boundary is reachable from this base and reports how far each traced point sits off-plan.
    def check_shape(self, shape, z_top):
        try:
            plan = self.plan_draw_shape(shape, z_top)
        except ValueError as exc:
            print(f'  {exc}')
            return False
        traj = plan[2][1]
        pts_xy = shape.boundary_points(self.ARC_POINTS)
        err = [1000 * np.linalg.norm(self.fk(q).t[:2] - np.array([x, y]))
            for q, (x, y) in zip(traj, pts_xy)]
        print(f'  StaubliShapeDraw: {shape.kind} traced, {len(traj)} steps, '
            f'off-plan {min(err):.1f}-{max(err):.1f} mm')
        return True