"""UR3e drawing in Swift, using the current teammate task/sensing/safety files.

    python ur3e_simulation.py --shape circle
    python ur3e_simulation.py --shape random --check
    python ur3e_simulation.py --image drawing.png

Shared planning lives here; ur3e_physical.py only adds the ROS connection.
No workcell GUI, crate, light curtain, tick loop or recovery is imported.
Importing this file does not open Swift or command a physical robot.
"""
import argparse
from dataclasses import dataclass

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from ir_support.robots import UR3e

from staubli_tx2_60_task import ShapeSpec, StaubliShapeDraw
from safety import (CollisionSafety, _joint_positions, _valid_links,
                    _segment_segment_distance)

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
PEN_CLEARANCE = 0.046  # Existing link-to-pen approximation, metres.
_collision = CollisionSafety()


@dataclass
class UR3eConfig:
    # Simulation defaults only. Physical CLI requires measured tool/surface values.
    centre: tuple = (-0.300, -0.130)
    surface_z: float = 0.090
    pen_length: float = 0.100
    size: float = 0.030
    approach: float = 0.100
    dt: float = 0.02
    draw_speed: float = 0.030
    move_speed: float = 0.050
    qd_limit_deg: float = 20.0
    home_deg: tuple = (45.0, -90.0, 90.0, -90.0, -90.0, 0.0)

    def __post_init__(self):
        positive = [self.size, self.approach, self.dt,
                    self.draw_speed, self.move_speed, self.qd_limit_deg]
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError('Size, approach, dt and speeds must be positive')
        if not np.isfinite(self.pen_length) or self.pen_length < 0:
            raise ValueError('Tool length must be non-negative; zero means flange only')
        if (np.shape(self.centre) != (2,) or np.shape(self.home_deg) != (6,)
                or not np.all(np.isfinite([*self.centre, self.surface_z, *self.home_deg]))):
            raise ValueError('Invalid centre, surface height or home joints')


class UR3eShapeDraw(StaubliShapeDraw):
    """Keep teammate drawing methods; override only the model, IK and timing."""

    def __init__(self, cfg=None, q_home=None):
        self.cfg = cfg or UR3eConfig()
        self.robot = UR3e()
        self.robot.tool = SE3(0, 0, self.cfg.pen_length)
        self.HOME = np.array(np.deg2rad(self.cfg.home_deg) if q_home is None else q_home,
                             dtype=float)
        if self.HOME.shape != (6,) or not np.all(np.isfinite(self.HOME)):
            raise ValueError('Home must contain six finite joint angles in radians')
        self.robot.q = self.HOME.copy()
        self.APPROACH = self.cfg.approach

    def solve_ik(self, x, y, z, q_seed=None, attempts=None):
        goal = SE3(x, y, z) * SE3.Ry(np.pi)
        seed = self.q.copy() if q_seed is None else np.asarray(q_seed, dtype=float)
        for k in range(self.IK_ATTEMPTS if attempts is None else attempts):
            initial = seed if k == 0 else seed + np.random.uniform(-0.3, 0.3, 6)
            sol = self.robot.ikine_LM(goal, q0=initial, joint_limits=True, tol=1e-10)
            if not sol.success:
                continue
            q = np.asarray(sol.q, dtype=float)
            q -= 2 * np.pi * np.round((q - seed) / (2 * np.pi))
            pose = self.fk(q)
            if (self.within_limits(q) and np.linalg.norm(pose.t - goal.t) < 0.00005
                    and np.linalg.norm(pose.R - goal.R) < 1e-4):
                return q
        return None

    def joint_traj(self, q_goal, steps=None, q_start=None):
        start = self.q.copy() if q_start is None else np.asarray(q_start, dtype=float)
        goal = np.asarray(q_goal, dtype=float)
        if (goal.shape != (6,) or not np.all(np.isfinite(goal))
                or not self.within_limits(goal)):
            raise ValueError('UR3e: unreachable or invalid joint target')
        # Quintic peak speed / mean speed = 1.875; timing is refined below.
        span = np.max(np.abs(goal - start))
        count = max(20, int(np.ceil(1.875 * span /
                    (np.deg2rad(self.cfg.qd_limit_deg) * self.cfg.dt))) + 1)
        return rtb.jtraj(start, goal, max(count, steps or 0)).q


def make_strokes(cfg, shape='circle', image=None):
    if image:
        import sensing
        strokes = sensing.strokes_from_jpeg(image, target_size=cfg.size)
        for stroke in strokes:
            stroke.cx, stroke.cy = cfg.centre
        return strokes
    return [ShapeSpec(*cfg.centre, cfg.size, None if shape == 'random' else shape)]


# UR3e preflight: teammate self-collision plus the existing pen check.
def self_collision(robot, q):
    result = _collision.self_collision(robot, q)
    if not result['safe'] or result.get('scale', 1.0) <= 0:
        return result['reason']
    points = _joint_positions(robot, q)
    if not np.all(np.isfinite(points)):
        raise ValueError('Non-finite robot geometry')
    tool = np.asarray(robot.tool.A)
    if np.linalg.norm(tool[:3, 3]) > 1e-9:
        flange = robot.fkine_all(q)[-1]
        tip = (flange.A @ tool[:, 3])[:3]
        # Ignore the directly attached wrist segment, as for connected links.
        for k in _valid_links(points)[:-1]:
            if _segment_segment_distance(points[k], points[k+1], flange.t, tip) <= PEN_CLEARANCE:
                return f'pen vs link {k+1}'
    return None


def check_self_collision_path(robot, frames, max_step_deg=0.5):
    """Raise before execution if any sampled pose self-collides.

    Includes the first pose and subdivisions of EVERY joint-space interval.
    Sampling is an approximation, not a continuous collision guarantee.
    """
    frames = np.asarray(frames, dtype=float)
    if (frames.ndim != 2 or frames.shape[1] != robot.n or len(frames) == 0
            or not np.all(np.isfinite(frames))):
        raise ValueError('Expected a finite, non-empty (N, robot.n) trajectory')
    if not np.isfinite(max_step_deg) or max_step_deg <= 0:
        raise ValueError('Collision sampling step must be positive')
    previous = frames[0]
    for k, goal in enumerate(frames):
        steps = max(1, int(np.ceil(np.max(np.abs(goal - previous))
                                   / np.deg2rad(max_step_deg))))
        for fraction in np.linspace(0, 1, steps + 1)[1:] if k else [0.0]:
            hit = self_collision(robot, previous + fraction * (goal - previous))
            if hit:
                raise ValueError(f'Self-collision at interval {k}: {hit}')
        previous = goal


def validate_frames(drawer, frames, dt=None):
    """Shared simulation/physical checks; returns an independent joint array."""
    dt = drawer.cfg.dt if dt is None else dt
    frames = np.array(frames, dtype=float, copy=True)
    if (frames.ndim != 2 or frames.shape[1] != drawer.robot.n or len(frames) == 0
            or not np.all(np.isfinite(frames)) or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Expected finite (N, robot.n) joints and positive dt')
    if any(not drawer.within_limits(q) for q in frames):
        raise ValueError('Joint limit exceeded')
    if len(frames) > 1:
        speed = np.max(np.abs(np.diff(frames, axis=0))) / dt
        if speed > np.deg2rad(drawer.cfg.qd_limit_deg) + 1e-9:
            raise ValueError('Joint speed limit exceeded; re-plan with valid timing')
    return frames


def prepare_plan(drawer, strokes):
    """Check start, reuse Staubli strokes, retime, then preflight the full task."""
    if not strokes:
        raise ValueError('No drawing strokes')
    cfg = drawer.cfg
    q_start = drawer.q.copy()
    validate_frames(drawer, q_start[None, :])
    check_self_collision_path(drawer.robot, q_start[None, :])
    raw = drawer.plan_draw_strokes(strokes, cfg.surface_z)
    raw.append(('home', drawer.joint_traj(q_start, q_start=raw[-1][1][-1])))
    plan = []
    previous = q_start
    for label, traj in raw:
        speed = cfg.draw_speed if label.endswith('_draw') else cfg.move_speed
        frames = []
        for goal in np.asarray(traj, dtype=float):
            if goal.shape != (6,) or not np.all(np.isfinite(goal)) or not drawer.within_limits(goal):
                raise ValueError(f'{label}: invalid joint pose or joint limit exceeded')
            delta = np.max(np.abs(goal - previous))
            if label.endswith('_draw') and delta > np.deg2rad(30):
                raise ValueError(f'{label}: IK branch jump; choose a different starting pose')
            if delta < 1e-12:
                continue
            distance = np.linalg.norm(drawer.fk(goal).t - drawer.fk(previous).t)
            count = max(1, int(np.ceil(max(delta / np.deg2rad(cfg.qd_limit_deg),
                                          distance / speed) / cfg.dt)))
            frames.extend(np.linspace(previous, goal, count + 1)[1:])
            previous = goal.copy()
        if frames:
            plan.append((label, np.asarray(frames)))
    frames = validate_frames(drawer, np.vstack([q_start[None, :]] + [traj for _, traj in plan]))
    check_self_collision_path(drawer.robot, frames)
    print(f'[UR3e] Self-collision check passed: {len(frames)} poses, '
          f'{(len(frames) - 1) * cfg.dt:.1f} s nominal duration')
    return plan, frames


def make_parser(physical=False):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--shape', choices=['random', 'circle', 'square', 'triangle'], default='circle')
    choice.add_argument('--image', help='Local PNG/JPEG image; uses teammate sensing.py')
    parser.add_argument('--centre', nargs=2, type=float, required=physical,
                        default=UR3eConfig.centre, metavar=('X', 'Y'),
                        help='Drawing centre in UR3e base coordinates, metres')
    parser.add_argument('--surface-z', type=float, required=physical,
                        default=UR3eConfig.surface_z, help='Measured lid height in base frame, metres')
    parser.add_argument('--pen-length', type=float, required=physical,
                        default=UR3eConfig.pen_length, help='Flange-to-tip along tool +Z, metres')
    parser.add_argument('--size', type=float, default=UR3eConfig.size, help='Shape half-size, metres')
    parser.add_argument('--check', action='store_true', help='Plan/check only; do not execute')
    return parser


def config_from_args(args):
    return UR3eConfig(centre=tuple(args.centre), surface_z=args.surface_z,
                     pen_length=args.pen_length, size=args.size)


def run_simulation(drawer, plan):
    """Execute the already checked plan with simple for loops; no GUI controls."""
    import swift
    from spatialgeometry import Cuboid, Cylinder, Sphere

    cfg = drawer.cfg
    env = swift.Swift()
    env.launch(realtime=True)
    try:
        drawer.add_to_env(env)
        env.add(Cuboid(scale=[1.20, 0.90, 0.04], pose=SE3(-0.25, 0, -0.02),
                       color=(0.62, 0.55, 0.45, 1)))
        # This slab represents the drawing surface at the configured height.
        env.add(Cuboid(scale=[0.12, 0.12, 0.01],
                       pose=SE3(*cfg.centre, cfg.surface_z - 0.005),
                       color=(0.76, 0.60, 0.38, 1)))
        pen = Cylinder(radius=0.006, length=cfg.pen_length, color=(0.1, 0.1, 0.12, 1))
        env.add(pen)
        pen.T = (drawer.fk() * SE3(0, 0, -cfg.pen_length / 2)).A
        for label, traj in plan:
            print(f'[SIM] {label}')
            last_mark = None
            for q in traj:
                drawer.robot.q = q
                pose = drawer.fk()
                pen.T = (pose * SE3(0, 0, -cfg.pen_length / 2)).A
                if label.endswith('_draw') and (last_mark is None or
                        np.linalg.norm(pose.t - last_mark) >= 0.001):
                    env.add(Sphere(radius=0.0012, pose=SE3(*pose.t), color=(0.1, 0.1, 0.12, 1)))
                    last_mark = pose.t.copy()
                env.step(cfg.dt)
        print('[SIM] Done. Ctrl+C to close Swift.')
        env.hold()
    finally:
        env.close()


def main(argv=None):
    args = make_parser().parse_args(argv)
    cfg = config_from_args(args)
    drawer = UR3eShapeDraw(cfg)
    plan, _ = prepare_plan(drawer, make_strokes(cfg, args.shape, args.image))
    if args.check:
        print('[SIM] Check only; Swift was not launched.')
        return
    run_simulation(drawer, plan)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[SIM] Closed')
