# Including libraries
import time
import numpy as np
import roboticstoolbox as rtb
import swift
from spatialmath import SE3
from spatialmath.base import trnorm
from spatialgeometry import Sphere

from staubli_tx2_60_task import ShapeSpec
from workcell import (
    BOX, BOX_TOP, BOX_HALF, BELT_Z,
    STORAGE_XY, FINISH_XY, STAND_TOP,
    Y_LOAD, Y_MARK, Y_PAINT,
    SHAPE_SIZE, INK_EVERY, PAINT_MARGIN,
    INK,
    MarkedBox, MarkPool, make_arms, build_workcell, make_stamp_shape,
)

def play(env, arm, traj, box=None, grasp=None, dt=0.02, on_frame=None):
    for k, q in enumerate(traj):
        arm.robot.q = q
        if grasp is not None and box is not None:
            box.set_pose(arm.fk() * grasp)
        if on_frame is not None:
            on_frame(arm, k)
        env.step(dt)

def run_pick(env, arm, box, x, y, z_top, dt=0.02):
    grasp = None
    for label, traj in arm.plan_pick(x, y, z_top):
        play(env, arm, traj, box, grasp, dt)
        if label == 'descend':
            grasp = SE3(trnorm((arm.fk().inv() * box.pose).A), check=False)
    return grasp

def run_place(env, arm, box, grasp, x, y, z_top, dt=0.02):
    for label, traj in arm.plan_place(x, y, z_top):
        play(env, arm, traj, box, grasp, dt)
        if label == 'descend':
            grasp = None
    return grasp

def run_plan(env, arm, plan, dt=0.02, marking=None):
    for label, traj in plan:
        hook = marking[1] if marking and label == marking[0] else None
        play(env, arm, traj, dt=dt, on_frame=hook)

def run_stamp(env, arm, box, shape, z_top, z_lift=0.0006, dt=0.02):
    plan = arm.plan_stamp(shape, z_top)
    stamped = False
    for label, traj in plan:
        play(env, arm, traj, dt=dt)
        if label == 'descend' and not stamped:
            point, on_lid = arm.lid_contact(z_top, half_width=BOX_HALF)
            if on_lid:
                item = make_stamp_shape(shape)
                env.add(item)
                box.add_mark(item, (point[0], point[1], z_top + z_lift))
            stamped = True
    return stamped

def convey(env, box, y_from, y_to, surface_z=BELT_Z, steps=90, dt=0.02):
    z = surface_z + BOX[2] / 2
    for y in rtb.trapezoidal(y_from, y_to, steps).q:
        box.set_pose(SE3(0.0, float(y), z))
        env.step(dt)

def make_marker(env, box, pool, every, lid_z, z_lift, max_radius=None, centre=None):
    def on_frame(arm, k):
        if k % every:
            return
        point, on_lid = arm.lid_contact(lid_z, half_width=BOX_HALF)
        if not on_lid:
            return
        if max_radius is not None:
            cx, cy = centre if centre is not None else box.pose.t[:2]
            if np.hypot(point[0] - cx, point[1] - cy) > max_radius:
                return
        item = pool.take()
        if item is not None:
            box.add_mark(item, (point[0], point[1], lid_z + z_lift))
    return on_frame

def run_cycle(env, arms, box, ink_pool, dt=0.02):
    arm1, arm2, arm3 = arms

    print('1/7  Kawasaki picks a box out of storage')
    grasp = run_pick(env, arm1, box, *STORAGE_XY, STAND_TOP + BOX[2], dt)

    print('2/7  Kawasaki sets it on the belt')
    grasp = run_place(env, arm1, box, grasp, 0.0, Y_LOAD, BOX_TOP, dt)
    run_plan(env, arm1, arm1.plan_home(), dt)

    print('3/7  belt carries it to the marking station')
    convey(env, box, Y_LOAD, Y_MARK, dt=dt)

    draw_shape = ShapeSpec(cx=0.0, cy=Y_MARK, size=SHAPE_SIZE)
    print(f'4/7  Staubli draws a {draw_shape.kind} on the lid')
    ink = make_marker(env, box, ink_pool, INK_EVERY, BOX_TOP, 0.0016)
    run_plan(env, arm2, arm2.plan_draw_shape(draw_shape, BOX_TOP), dt, ('draw', ink))
    run_plan(env, arm2, arm2.plan_home(), dt)
    print(f'     {ink_pool.used} ink marks laid')

    print('5/7  belt carries it to the paint station')
    convey(env, box, Y_MARK, Y_PAINT, dt=dt)

    fill_shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=draw_shape.size, kind=draw_shape.kind)
    print(f'6/7  Nachi stamps the {fill_shape.kind}')
    stamped = run_stamp(env, arm3, box, fill_shape, BOX_TOP, dt=dt)
    run_plan(env, arm3, arm3.plan_home(), dt)
    print(f'     stamp placed on the {fill_shape.kind}' if stamped else '     stamp missed the lid')

    print('7/7  belt returns it, Kawasaki moves it to the finish area')
    convey(env, box, Y_PAINT, Y_LOAD, steps=150, dt=dt)
    grasp = run_pick(env, arm1, box, 0.0, Y_LOAD, BOX_TOP, dt)
    run_place(env, arm1, box, grasp, *FINISH_XY, STAND_TOP + BOX[2], dt)
    run_plan(env, arm1, arm1.plan_home(), dt)
    print('cycle complete')

def check_layout():
    print('\nLayout check')
    arm1, arm2, arm3 = make_arms()
    ok = arm1.check_stations([
        ('storage', *STORAGE_XY, STAND_TOP + BOX[2]),
        ('belt', 0.0, Y_LOAD, BOX_TOP),
        ('finish', *FINISH_XY, STAND_TOP + BOX[2])])
    for kind in ['circle', 'square', 'triangle']:
        draw_shape = ShapeSpec(cx=0.0, cy=Y_MARK, size=SHAPE_SIZE, kind=kind)
        fill_shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=SHAPE_SIZE - PAINT_MARGIN, kind=kind)
        ok &= arm2.check_shape(draw_shape, BOX_TOP)
        ok &= arm3.check_stamp(fill_shape, BOX_TOP)
    print('  layout OK\n' if ok else '  LAYOUT PROBLEM\n')
    return ok


if __name__ == "__main__":
    check_layout()

    env = swift.Swift()
    env.launch(realtime=True)

    build_workcell(env)
    arms = make_arms()
    for arm in arms:
        arm.add_to_env(env)

    box = MarkedBox(env, *STORAGE_XY, STAND_TOP)
    ink_pool = MarkPool(env, 140, lambda: Sphere(radius=0.0018, color=INK))

    env.set_camera_pose([2.9, -2.6, 2.1], [0, 0, 0.8])
    env.step(0.02)

    run_cycle(env, arms, box, ink_pool)

    env.hold()
    time.sleep(3)