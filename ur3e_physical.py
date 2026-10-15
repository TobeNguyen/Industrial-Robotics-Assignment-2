#!/usr/bin/env python3
import time

import numpy as np
import roslibpy
from spatialmath import SE3

from ur3e_simulation import (
    JOINT_NAMES, UR3eConfig, UR3eShapeDraw, make_strokes, prepare_plan,
    make_parser, config_from_args, validate_frames, check_self_collision_path,
)

ROS_HOST, ROS_PORT = '192.168.27.1', 9090
STATE_TOPIC = '/ur/joint_states'
COMMAND_TOPIC = '/ur/scaled_pos_joint_traj_controller/command'
PROGRAM_TOPIC = '/ur/ur_hardware_interface/robot_program_running'
FEEDBACK_MAX_AGE = 0.5
START_TOLERANCE = np.deg2rad(0.5)
PRE_SEND_TOLERANCE = np.deg2rad(0.05)

# Relaxed, non-singular pose (base, shoulder, elbow, wrist1, wrist2, wrist3), degrees.
SAFE_POSE_DEG = [0.0, -100.0, -90.0, -80.0, 90.0, 0.0]
MIN_MANIPULABILITY = 0.002      # Below this a pose is treated as near-singular.
SAFE_MOVE_SPEED_DEG = 8.0       # Joint speed (deg/s) for the move to the safe pose.
SAFE_POSE_TOLERANCE = np.deg2rad(1.0)   # Max joint error accepted after the move.
ALREADY_THERE_TOLERANCE = np.deg2rad(2.0)  # Skip the move if the robot is this close.

current_pos = None  # One atomic snapshot: (ordered positions, local receipt time)


def _rotation_error(R_goal, R_now):
    """Orientation error as a base-frame rotation vector (small-angle form)."""
    R = R_goal @ R_now.T
    return 0.5 * np.array([R[2, 1] - R[1, 2],
                           R[0, 2] - R[2, 0],
                           R[1, 0] - R[0, 1]])


def solve_ik_dls(drawer, goal, seed, max_iter=200, damping=0.02,
                 tol_pos=1e-7, tol_rot=1e-7, max_joint_step=np.deg2rad(5)):
    """Damped least-squares IK started from `seed`.

    Stays on the seed's IK branch and tolerates configurations that are close
    to a singularity. Returns joint angles, or None if it did not converge.
    """
    try:
        q = np.array(seed, dtype=float, copy=True)
        goal_A = goal.A
        for _ in range(max_iter):
            A = drawer.fk(q).A
            e = np.r_[goal_A[:3, 3] - A[:3, 3],
                      _rotation_error(goal_A[:3, :3], A[:3, :3])]
            if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
                return q
            J = np.asarray(drawer.robot.jacob0(q), dtype=float)
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            biggest = np.max(np.abs(dq))
            if biggest > max_joint_step:
                dq *= max_joint_step / biggest
            q = q + dq
    except Exception:
        return None
    return None


def solve_ik(drawer, goal, seed):
    """Try DLS first, then fall back to Levenberg-Marquardt with looser settings."""
    q = solve_ik_dls(drawer, goal, seed)
    if q is not None:
        return q
    attempts = (
        {'joint_limits': True, 'tol': 1e-10},
        {'joint_limits': False, 'tol': 1e-8, 'ilimit': 200, 'slimit': 20},
    )
    for kwargs in attempts:
        try:
            sol = drawer.robot.ikine_LM(goal, q0=seed, **kwargs)
        except Exception:
            continue
        if sol.success:
            return np.array(sol.q, dtype=float, copy=True)
    return None


def manipulability(drawer, q):
    """Yoshikawa manipulability index (near 0 means close to a singularity)."""
    J = np.asarray(drawer.robot.jacob0(q), dtype=float)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def describe_pose(drawer, q):
    """Return a short diagnostic string for a joint configuration."""
    text = 'joints (deg): ' + np.array2string(np.rad2deg(q), precision=1, suppress_small=True)
    try:
        text += f'; manipulability: {manipulability(drawer, q):.5f} (near 0 means close to a singularity)'
    except Exception:
        pass
    try:
        text += f'; within limits: {bool(drawer.within_limits(q))}'
    except Exception:
        pass
    return text


def prepare_air_plan(drawer, strokes):
    """Translate the path so its first point is the current flange position.

    Keep orientation, no lift/touch/retract or move to an arbitrary home.
    Between image strokes, travel straight in the same plane (no physical ink).
    Checks concern the robot model, not unknown obstacles in the real cell.
    """
    q_start = drawer.q.copy()
    validate_frames(drawer, q_start[None, :])
    check_self_collision_path(drawer.robot, q_start[None, :])
    if not strokes:
        raise ValueError('No air strokes')
    paths = [np.asarray(s.boundary_points(), dtype=float) for s in strokes]
    if any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or
           not np.all(np.isfinite(p)) for p in paths):
        raise ValueError('Expected finite 2D stroke points')
    pose = drawer.fk(q_start)
    anchor = paths[0][0]
    offsets = np.vstack(paths) - anchor
    if np.max(np.linalg.norm(offsets, axis=1)) > 0.030 + 1e-9:
        raise ValueError('Air path exceeds 30 mm from start; reduce --size')
    print('[AIR] Start ' + describe_pose(drawer, q_start))
    targets = [pose.t + np.r_[xy, 0.] for xy in offsets]
    targets.append(pose.t.copy())
    frames = [q_start]
    previous_xyz = pose.t.copy()
    for target_index, target in enumerate(targets):
        distance = np.linalg.norm(target - previous_xyz)
        count = max(1, int(np.ceil(distance / (drawer.cfg.draw_speed * drawer.cfg.dt))))
        for xyz in np.linspace(previous_xyz, target, count + 1)[1:]:
            if np.linalg.norm(xyz - drawer.fk(frames[-1]).t) < 1e-9:
                continue
            matrix = pose.A.copy()
            matrix[:3, 3] = xyz
            goal = SE3(matrix)
            seed = frames[-1]
            q = solve_ik(drawer, goal, seed)
            if q is None:
                raise ValueError(
                    'Air path unreachable from current pose; choose another clear start pose.\n'
                    f'  Failed at target {target_index + 1}/{len(targets)}, '
                    f'offset from start (mm): {np.round((xyz - pose.t) * 1000, 2)}\n'
                    f'  Last good {describe_pose(drawer, seed)}\n'
                    '  Hint: try a different --safe-pose (elbow bent, wrist pointing down, '
                    'away from joint limits).')
            q = np.array(q, dtype=float, copy=True)
            q -= 2 * np.pi * np.round((q - seed) / (2 * np.pi))
            actual = drawer.fk(q)
            if (not drawer.within_limits(q) or not np.all(np.isfinite(q)) or
                    np.linalg.norm(actual.t - xyz) > 0.00005 or
                    np.linalg.norm(actual.R - pose.R) > 1e-4):
                raise ValueError('Air path IK failed pose/limit verification')
            delta = np.max(np.abs(q - seed))
            if delta > np.deg2rad(5):
                raise ValueError('Air path IK branch jump; choose another clear start pose')
            steps = max(1, int(np.ceil(delta / (np.deg2rad(drawer.cfg.qd_limit_deg) * drawer.cfg.dt))))
            for sample in np.linspace(seed, q, steps + 1)[1:]:
                tcp = drawer.fk(sample)
                if (np.linalg.norm(tcp.t - xyz) > 0.0005 or
                        np.linalg.norm(tcp.R - pose.R) > 1e-3):
                    raise ValueError('Air path interpolation deviates; choose another start pose')
                frames.append(sample)
        previous_xyz = target
    frames = validate_frames(drawer, frames)
    check_self_collision_path(drawer.robot, frames)
    print(f'[AIR] Flange start (m): {pose.t}; {len(frames)} points; '
          f'{(len(frames)-1)*drawer.cfg.dt:.1f} s nominal. No obstacle sensing.')
    return frames


def joint_state_cb(message):
    global current_pos
    try:
        positions = message['position']
        names = message.get('name', [])
        if names:
            if len(names) != len(positions) or len(set(names)) != len(names):
                raise ValueError('Invalid joint names')
            q = np.array([positions[names.index(name)] for name in JOINT_NAMES], dtype=float)
        else:
            if len(positions) != 6:
                raise ValueError('Expected six joints')
            q = np.asarray(positions, dtype=float)[[2, 1, 0, 3, 4, 5]]
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError('Invalid joint positions')
        current_pos = (q.copy(), time.monotonic())
    except (KeyError, TypeError, ValueError, IndexError):
        current_pos = None  # Invalid feedback must not leave a valid stale pose behind.


def wait_for_joint_state(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = current_pos
        if snapshot is not None and time.monotonic() - snapshot[1] < FEEDBACK_MAX_AGE:
            return snapshot[0].copy()
        time.sleep(0.05)
    raise RuntimeError('No fresh valid joint state received from /ur/joint_states')


def wait_for_program_status(program, timeout=3.0):
    """Give the bridge a moment to deliver the first robot_program_running message."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and program.get('data') is None:
        time.sleep(0.05)


def trajectory_message(frames, dt):
    frames = np.asarray(frames, dtype=float)
    if (frames.ndim != 2 or frames.shape[1] != 6 or len(frames) < 2
            or not np.all(np.isfinite(frames)) or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Expected at least two finite six-joint poses and positive dt')
    points = []
    previous = -1
    for i, q in enumerate(frames):
        nanos = round(i * dt * 1_000_000_000)
        if nanos <= previous:
            raise ValueError('Trajectory times must increase at nanosecond resolution')
        secs, nsecs = divmod(nanos, 1_000_000_000)
        points.append({'positions': q.tolist(),
                       'time_from_start': {'secs': secs, 'nsecs': nsecs}})
        previous = nanos
    return {'joint_names': JOINT_NAMES.copy(), 'points': points}


def move_ur_joint_positions(client, joint_positions, drawer, dt=None, before_publish=None):
    """Publish the checked FULL task, not just its final joint target.

    joint_positions is now (N, 6), radians. The caller owns the subscription.
    This replaces the supplied script's two-point move/duration API.
    """
    dt = drawer.cfg.dt if dt is None else dt
    frames = validate_frames(drawer, joint_positions, dt)
    if len(frames) < 2:
        raise ValueError('Expected at least two trajectory poses')
    q_now = wait_for_joint_state()
    if np.max(np.abs(q_now - frames[0])) > START_TOLERANCE:
        raise RuntimeError('Robot moved during planning; rerun from the new pose')
    count = max(1, int(np.ceil(np.max(np.abs(frames[0] - q_now)) /
                               (np.deg2rad(drawer.cfg.qd_limit_deg) * dt))))
    connector = np.linspace(q_now, frames[0], count + 1)[:-1]
    frames = validate_frames(drawer, np.vstack([connector, frames]), dt)
    check_self_collision_path(drawer.robot, frames)
    message = trajectory_message(frames, dt)
    topic = roslibpy.Topic(client, COMMAND_TOPIC, 'trajectory_msgs/JointTrajectory')
    try:
        topic.advertise()
        latest = wait_for_joint_state()
        if np.max(np.abs(latest - q_now)) > PRE_SEND_TOLERANCE:
            raise RuntimeError('Robot moved during checking; rerun the task')
        if before_publish is not None:
            before_publish()
        topic.publish(roslibpy.Message(message))
        duration = (len(frames) - 1) * dt
        print(f'[ROS] Published {len(frames)} points ({duration:.1f} s nominal).')
        time.sleep(duration + 1.0)  # Same wait pattern as the supplied working script.
        print('[ROS] Nominal wait finished; the topic provides no task-completion result.')
    finally:
        topic.unadvertise()


def build_safe_pose_frames(drawer, q_start, q_safe):
    """Joint-space path from the live pose to the safe pose, speed limited and checked."""
    speed_deg = min(SAFE_MOVE_SPEED_DEG, 0.8 * drawer.cfg.qd_limit_deg)
    delta = np.max(np.abs(q_safe - q_start))
    steps = max(2, int(np.ceil(delta / (np.deg2rad(speed_deg) * drawer.cfg.dt))))
    frames = np.linspace(q_start, q_safe, steps + 1)
    frames = validate_frames(drawer, frames, drawer.cfg.dt)
    check_self_collision_path(drawer.robot, frames)
    return frames


def confirm(prompt, assume_yes):
    if assume_yes:
        return True
    return input(prompt).strip().upper() == 'YES'


def main(argv=None):
    global current_pos
    parser = make_parser()
    parser.set_defaults(centre=None, surface_z=None, pen_length=None, size=0.010)
    parser.description = __doc__
    parser.add_argument('--host', default=ROS_HOST)
    parser.add_argument('--port', type=int, default=ROS_PORT)
    parser.add_argument('--skip-safe-pose', action='store_true',
                        help='Do not move to the safe pose first; plan from the live pose')
    parser.add_argument('--safe-pose', type=float, nargs=6, default=SAFE_POSE_DEG,
                        metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                        help='Safe start pose in degrees (base, shoulder, elbow, wrist1-3)')
    parser.add_argument('--yes', action='store_true',
                        help='Skip the YES confirmations before robot motion')
    args = parser.parse_args(argv)
    geometry = [args.centre is not None, args.surface_z is not None, args.pen_length is not None]
    if any(geometry) and not all(geometry):
        parser.error('For surface mode provide --centre, --surface-z and --pen-length together; omit all for air mode')
    air = not any(geometry)
    cfg = (UR3eConfig(centre=(0., 0.), surface_z=0., pen_length=0., size=args.size,
                     draw_speed=0.010, move_speed=0.010, qd_limit_deg=10.)
           if air else config_from_args(args))
    strokes = make_strokes(cfg, args.shape, args.image)
    client = roslibpy.Ros(host=args.host, port=args.port)
    listener = None
    program_listener = None
    program = {'data': None}

    def require_program():
        if program.get('data') is not True:
            raise RuntimeError('External Control is not running (or no status received). No trajectory sent.')

    current_pos = None
    try:
        client.run()
        listener = roslibpy.Topic(client, STATE_TOPIC, 'sensor_msgs/JointState')
        listener.subscribe(joint_state_cb)
        program_listener = roslibpy.Topic(client, PROGRAM_TOPIC, 'std_msgs/Bool')
        program_listener.subscribe(lambda msg: program.update(data=msg.get('data')))
        print('[ROS] Waiting for current joint state...')
        q_start = wait_for_joint_state()
        drawer = UR3eShapeDraw(cfg, q_home=q_start)
        print('[POSE] Live ' + describe_pose(drawer, q_start))

        if not args.skip_safe_pose:
            q_safe = np.deg2rad(np.asarray(args.safe_pose, dtype=float))
            if not drawer.within_limits(q_safe):
                raise ValueError('Safe pose is outside the joint limits; choose another --safe-pose')
            if manipulability(drawer, q_safe) < MIN_MANIPULABILITY:
                raise ValueError('Safe pose is itself near a singularity; choose another --safe-pose')
            if np.max(np.abs(q_start - q_safe)) <= ALREADY_THERE_TOLERANCE:
                print('[POSE] Robot is already at the safe pose; no move needed.')
            else:
                print('[POSE] Target safe pose (deg): '
                      + np.array2string(np.rad2deg(q_safe), precision=1, suppress_small=True))
                safe_frames = build_safe_pose_frames(drawer, q_start, q_safe)
                print(f'[POSE] Safe-pose path checked: {len(safe_frames)} points, '
                      f'{(len(safe_frames) - 1) * drawer.cfg.dt:.1f} s nominal.')
                if args.check:
                    print('[POSE] Check only; robot not moved. Planning the task from the safe pose.')
                    drawer = UR3eShapeDraw(cfg, q_home=q_safe)
                else:
                    wait_for_program_status(program)
                    require_program()
                    if not confirm('[POSE] Robot will MOVE to the safe pose. Clear the area, keep a hand '
                                   'near the E-stop, then type YES to continue: ', args.yes):
                        print('[POSE] Cancelled by user. Nothing was sent.')
                        return
                    move_ur_joint_positions(client, safe_frames, drawer, before_publish=require_program)
                    q_after = wait_for_joint_state()
                    if np.max(np.abs(q_after - q_safe)) > SAFE_POSE_TOLERANCE:
                        raise RuntimeError('Robot did not reach the safe pose; check the pendant and rerun')
                    print('[POSE] Safe pose reached.')
                    drawer = UR3eShapeDraw(cfg, q_home=q_after)

        if air:
            print('[AIR] Pen removed; flange path at current height/orientation, no gripper commands.')
            frames = prepare_air_plan(drawer, strokes)
        else:
            _, frames = prepare_plan(drawer, strokes)
        if args.check:
            print('[ROS] Check only; no trajectory published.')
            return

        wait_for_program_status(program)
        require_program()
        if not confirm('[ROS] Robot will MOVE along the planned path. Type YES to continue: ', args.yes):
            print('[ROS] Cancelled by user. Nothing was sent.')
            return
        move_ur_joint_positions(client, frames, drawer, before_publish=require_program)
    finally:
        try:
            if program_listener is not None:
                program_listener.unsubscribe()
            if listener is not None:
                listener.unsubscribe()
        finally:
            client.terminate()
        print('[ROS] Finish')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[APP] Interrupted. Closing this script does not cancel a published robot trajectory.')

