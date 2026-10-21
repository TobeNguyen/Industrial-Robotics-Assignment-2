# Including libraries
import time

import cv2
import numpy as np
import roboticstoolbox as rtb
import swift
from spatialmath import SE3
from spatialmath.base import trnorm
from spatialgeometry import Sphere

import gui
import physical_estop
import sensing

from staubli_tx2_60_task import ShapeSpec
from workcell import (
    BOX, BOX_TOP, BOX_HALF, BELT_Z,
    STORAGE_XY, FINISH_XY, STAND_TOP,
    Y_LOAD, Y_MARK, Y_PAINT,
    SHAPE_SIZE, INK_EVERY, PAINT_MARGIN,
    INK, PAINTED,INK_RADIUS, INK_SPACING, INK_POOL_SIZE,
    MarkedBox, MarkPool, make_arms, build_workcell, make_stamp_shape,
    make_camera_model,
)

from safety import RobotSafetyController

COLOR_MAP = {
    'red':    (0.80, 0.15, 0.15, 1),
    'blue':   (0.20, 0.45, 0.78, 1),
    'green':  (0.20, 0.65, 0.30, 1),
    'yellow': (0.90, 0.80, 0.10, 1),
}

# Simulated RGB-D camera mounted on the Staubli wrist (replace with real camera frames and calibration on the real cell)
# The arm moves to a look pose, so the camera pose comes from arm2.cam_pose(), not from a fixed transform
# Keep the lid inside the image: 2 * BOX_HALF * fx / CAM_HEIGHT must be well below the image height
CAM_HEIGHT = 0.18               # camera height above the lid at the look pose (m); lower it if the look pose is unreachable
CAM_SIZE = (320, 240)           # (width, height) px
CAM_INTRINSICS = {'fx': 180.0, 'fy': 180.0, 'cx': 160.0, 'cy': 120.0}
SERVO_NUDGE = (0.004, 0.0)      # deliberate 4 mm box shift before the 2nd stroke, to show the correction; None to disable
SERVO_FRAMES = 5                # camera frames averaged by each measurement
SERVO_EVERY_STROKE = True       # True: go back to the look pose and re-measure before every stroke; False: measure once

# The robot never moves slower than 1/20 of the normal speed while it is only being slowed down (not stopped)
MIN_SCALE = 0.05

def resolve_color(value):
    # Accepts either an (r, g, b) tuple/list in 0-255 or a legacy colour name
    if isinstance(value, (tuple, list)) and len(value) == 3:
        r, g, b = (max(0, min(255, int(c))) / 255 for c in value)
        return (r, g, b, 1)
    if isinstance(value, str):
        return COLOR_MAP.get(value, PAINTED)
    return PAINTED

# Blocks (holding the position) while E-STOP / collision / severe singularity gives scale 0.
# Returns the speed scale (0 < scale <= 1). Raises CycleAborted if RESET PROCESS aborted the cycle.
def safe_scale(env, arm, q, controller, dt):
    while True:
        if gui.abort_requested():
            raise gui.CycleAborted()
        if controller is None:
            return 1.0
        scale = gui.safety_scale(arm, q, controller)
        if scale > 0.0:
            return scale
        gui.refresh_status_label()
        env.step(dt)

# Holds the whole cell while E-STOP is active (used between robot motions, e.g. by the belt)
def check_continue(env, dt=0.02):
    while True:
        if gui.abort_requested():
            raise gui.CycleAborted()
        gui.refresh_status_label()  # Check crate collisions before allowing the next motion.
        if not gui._estopped():
            return
        env.step(dt)

def play(env, arm, traj, box=None, grasp=None, dt=0.02, on_frame=None, controller=None):
    for k, q in enumerate(traj):
        scale = safe_scale(env, arm, q, controller, dt)
        step_dt = dt / max(scale, MIN_SCALE)        # lower speed scale -> larger time step

        check_continue(env, dt)  # Stop/reset may arrive during safety calculation.
        arm.robot.q = q
        cam_model = getattr(arm, 'cam_model', None)
        if cam_model is not None:
            cam_model.T = arm.cam_pose().A
        if grasp is not None and box is not None:
            box.set_pose(arm.fk() * grasp)
        if on_frame is not None:
            on_frame(arm, k)
        gui.refresh_status_label()
        env.step(step_dt)
    check_continue(env, dt)  # Also handle stop/reset on the final frame.
    return True

def run_pick(env, arm, box, x, y, z_top, dt=0.02, controller=None):
    grasp = None
    for label, traj in arm.plan_pick(x, y, z_top):
        play(env, arm, traj, box, grasp, dt, controller=controller)
        if label == 'descend':
            grasp = SE3(trnorm((arm.fk().inv() * box.pose).A), check=False)
    return grasp

def run_place(env, arm, box, grasp, x, y, z_top, dt=0.02, controller=None):
    for label, traj in arm.plan_place(x, y, z_top):
        play(env, arm, traj, box, grasp, dt, controller=controller)
        if label == 'descend':
            grasp = None
    return grasp

def run_plan(env, arm, plan, dt=0.02, marking=None, controller=None):
    for label, traj in plan:
        hook = marking[1] if marking and label == marking[0] else None
        play(env, arm, traj, dt=dt, on_frame=hook, controller=controller)

def run_stamp(env, arm, box, shape, z_top, color=PAINTED, z_lift=0.0006, dt=0.02, controller=None):
    plan = arm.plan_stamp(shape, z_top)
    stamped = False
    for label, traj in plan:
        play(env, arm, traj, dt=dt, controller=controller)
        if label == 'descend' and not stamped:
            point, on_lid = arm.lid_contact(z_top, half_width=BOX_HALF)
            if on_lid:
                item = make_stamp_shape(shape, color=color)
                env.add(item)
                box.add_mark(item, (point[0], point[1], z_top + z_lift))
            stamped = True
    return stamped

def convey(env, box, y_from, y_to, surface_z=BELT_Z, steps=90, dt=0.02):
    z = surface_z + BOX[2] / 2
    for y in rtb.trapezoidal(y_from, y_to, steps).q:
        check_continue(env, dt)
        box.set_pose(SE3(0.0, float(y), z))
        env.step(dt)

def make_marker(env, box, pool, every, lid_z, z_lift, max_radius=None, centre=None, min_spacing=0.0):
    last = []                           # world xy of the last dot laid (used by min_spacing)

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
        # dots are laid by distance, so short strokes do not waste dots and long ones are not cut short
        if min_spacing > 0.0 and last and np.hypot(point[0] - last[0], point[1] - last[1]) < min_spacing:
            return
        item = pool.take()
        if item is not None:
            box.add_mark(item, (point[0], point[1], lid_z + z_lift))
            last[:] = [point[0], point[1]]
    return on_frame

def build_draw_shape(command, cx, cy, size=SHAPE_SIZE):
    mode = command.get('mode')
    if mode == 'random':
        return ShapeSpec(cx=cx, cy=cy, size=size)
    if mode == 'basic':
        shape = command.get('shape')
        if shape is None:
            raise ValueError('build_draw_shape: basic mode requires a shape')
        return ShapeSpec(cx=cx, cy=cy, size=size, kind=shape)
    if mode == 'image':
        image_name = gui.get_status_snapshot().get('image_name')
        model = gui.image_models.get(image_name)
        if model is None:
            raise ValueError(f'build_draw_shape: no processed image model for {image_name!r}')
        model.cx, model.cy = cx, cy
        return model
    raise ValueError(f'build_draw_shape: unknown mode {mode!r}')

# Returns the list of strokes (outer outline first) for an imported image, or None when only one stroke is available
def get_draw_strokes(command, cx, cy):
    if command.get('mode') != 'image':
        return None
    image_name = gui.get_status_snapshot().get('image_name')
    strokes = gui.image_strokes.get(image_name)
    if not strokes:
        return None
    for s in strokes:
        s.cx, s.cy = cx, cy
    return strokes

# Synthetic RGB-D frame of the lid seen from the wrist camera (stands in for the real camera)
def simulate_rgbd(box, cam_pose):
    w, h = CAM_SIZE
    fx, fy = CAM_INTRINSICS['fx'], CAM_INTRINSICS['fy']
    u0, v0 = CAM_INTRINSICS['cx'], CAM_INTRINSICS['cy']
    bx, by = box.pose.t[0], box.pose.t[1]
    corners = [(bx + sx * BOX_HALF, by + sy * BOX_HALF)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    # lid corners expressed in the camera frame, then pinhole projection
    world_to_cam = cam_pose.inv()
    pts = [(world_to_cam * SE3(x, y, BOX_TOP)).t for x, y in corners]
    pix = np.array([[fx * p[0] / p[2] + u0, fy * p[1] / p[2] + v0] for p in pts])
    gray = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(gray, np.round(pix).astype(np.int32), 255)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    depth = np.full((h, w), float(np.mean([p[2] for p in pts])), dtype=float)
    return rgb, depth

# Pose of the workpiece (world frame) seen from the current camera pose, used as the target for every later servo step (None if not visible)
def capture_reference(box, cam_pose):
    rgb, depth = simulate_rgbd(box, cam_pose)
    try:
        state = sensing.get_workpiece_state(rgb, depth, CAM_INTRINSICS, cam_pose, (0.0, 0.0, 0.0))
    except ValueError:
        return None
    return state['actual_pose']

# Rotates a stroke by the correction and moves its centre by the correction's shift, so the lid centre follows the box
def corrected_stroke(stroke, correction):
    theta = np.arctan2(correction.R[1, 0], correction.R[0, 0])
    pts = sensing.transform_path(stroke.points, SE3.Rz(theta), 0.0)
    xy = [(x, y) for x, y, _ in pts]
    return sensing.create_shape_model(xy, cx=stroke.cx + correction.t[0],
                                    cy=stroke.cy + correction.t[1], size=stroke.size)

# Optional deliberate nudge, ONE measurement from the wrist camera, then every given stroke is corrected with the same result
def servo_strokes(env, box, strokes, reference, cam_pose, nudge=None):
    if nudge is not None:
        box.set_pose(SE3(nudge[0], nudge[1], 0.0) * box.pose)
        env.step(0.02)
    correction, state = measure_correction(box, reference, cam_pose)
    if state['error'] is None:
        gui.update_status(servo='camera not ready')
        gui.log_step('Visual servo: camera not ready, strokes not corrected')
        return list(strokes)
    dx, dy, dtheta = state['error']
    shift_mm = 1000 * np.hypot(dx, dy)
    if not state['applied']:
        gui.update_status(servo='on target')
        gui.log_step(f'Visual servo: on target ({shift_mm:.1f} mm)')
        return list(strokes)
    gui.update_status(servo=f'corrected {shift_mm:.1f} mm')
    gui.log_step(f'Visual servo: box off by {shift_mm:.1f} mm, {len(strokes)} stroke(s) corrected')
    return [corrected_stroke(s, correction) for s in strokes]

# Measures the box ONCE from several frames and returns (correction SE3, state). The lid is square, so the angle error is wrapped to +-45 deg
def measure_correction(box, reference, cam_pose, n_frames=SERVO_FRAMES):
    frames = [simulate_rgbd(box, cam_pose) for _ in range(n_frames)]
    state = sensing.measure_workpiece_state(frames, CAM_INTRINSICS, cam_pose, reference)
    if state['error'] is None or not state['applied']:
        return SE3(), state
    dx, dy, dtheta = state['error']
    dtheta = (dtheta + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return sensing.calculate_visual_correction(np.array([dx, dy, dtheta])), state

# controllers: list [Kawasaki, Staubli, Nachi] (same order as arms), or None to run without safety
def run_cycle(env, arms, box, ink_pool, dt=0.02, shape=None, color=PAINTED, strokes=None, controllers=None):
    arm1, arm2, arm3 = arms
    if controllers is None:
        controllers = [None, None, None]
    controller1, controller2, controller3 = controllers

    print('1/7  Kawasaki picks a box out of storage')
    grasp = run_pick(env, arm1, box, *STORAGE_XY, STAND_TOP + BOX[2], dt, controller=controller1)

    print('2/7  Kawasaki sets it on the belt')
    grasp = run_place(env, arm1, box, grasp, 0.0, Y_LOAD, BOX_TOP, dt, controller=controller1)
    run_plan(env, arm1, arm1.plan_home(), dt, controller=controller1)

    print('3/7  belt carries it to the marking station')
    convey(env, box, Y_LOAD, Y_MARK, dt=dt)

    draw_shape = shape if shape is not None else ShapeSpec(cx=0.0, cy=Y_MARK, size=SHAPE_SIZE)
    print(f'4/7  Staubli draws a {draw_shape.kind} on the lid')
    stroke_list = strokes if strokes else [draw_shape]
    multi = len(stroke_list) > 1
    raw_strokes = list(stroke_list)     # uncorrected strokes, every re-measurement corrects these against the reference
    reference = None
    if multi:
        # eye-in-hand: the arm goes to the look pose, then the wrist camera takes the reference frame
        run_plan(env, arm2, arm2.plan_look(0.0, Y_MARK, BOX_TOP, CAM_HEIGHT), dt, controller=controller2)
        reference = capture_reference(box, arm2.cam_pose())
        if reference is not None:
            # one measurement (with the deliberate nudge), the same correction is used for every stroke
            stroke_list = servo_strokes(env, box, stroke_list, reference, arm2.cam_pose(), SERVO_NUDGE)
    ink = make_marker(env, box, ink_pool, 1 if multi else INK_EVERY, BOX_TOP, 0.0016,
                    min_spacing=INK_SPACING if multi else 0.0)
    for i, stroke in enumerate(stroke_list):
        if multi:
            print(f'     stroke {i + 1}/{len(stroke_list)}')
            if SERVO_EVERY_STROKE and reference is not None and i > 0:
                # look again before this stroke, the correction is measured from the wrist camera
                run_plan(env, arm2, arm2.plan_look(0.0, Y_MARK, BOX_TOP, CAM_HEIGHT), dt, controller=controller2)
                stroke = servo_strokes(env, box, [raw_strokes[i]], reference, arm2.cam_pose())[0]
        run_plan(env, arm2, arm2.plan_draw_shape(stroke, BOX_TOP), dt, ('draw', ink), controller=controller2)
    run_plan(env, arm2, arm2.plan_home(), dt, controller=controller2)
    print(f'     {ink_pool.used} ink marks laid')
    items = getattr(ink_pool, 'items', None)
    if items and ink_pool.used >= len(items):
        gui.log_step('Ink pool used up: the last strokes may be incomplete (raise INK_POOL_SIZE)')

    print('5/7  belt carries it to the paint station')
    convey(env, box, Y_MARK, Y_PAINT, dt=dt)

    if type(draw_shape).__name__ == 'ShapeSpec':
        fill_shape = ShapeSpec(cx=0.0, cy=Y_PAINT, size=draw_shape.size, kind=draw_shape.kind)
    else:
        fill_shape = draw_shape
        fill_shape.cx, fill_shape.cy = 0.0, Y_PAINT
    print(f'6/7  Nachi stamps the {fill_shape.kind} in {color}')
    stamped = run_stamp(env, arm3, box, fill_shape, BOX_TOP, color=color, dt=dt, controller=controller3)
    run_plan(env, arm3, arm3.plan_home(), dt, controller=controller3)
    print(f'     stamp placed on the {fill_shape.kind}' if stamped else '     stamp missed the lid')

    print('7/7  belt returns it, Kawasaki moves it to the finish area')
    convey(env, box, Y_PAINT, Y_LOAD, steps=150, dt=dt)
    grasp = run_pick(env, arm1, box, 0.0, Y_LOAD, BOX_TOP, dt, controller=controller1)
    run_place(env, arm1, box, grasp, *FINISH_XY, STAND_TOP + BOX[2], dt, controller=controller1)
    run_plan(env, arm1, arm1.plan_home(), dt, controller=controller1)
    print('cycle complete')

def run_gui_cycle(env, arms, box, ink_pool, command, dt=0.02, controllers=None):
    try:
        draw_shape = build_draw_shape(command, cx=0.0, cy=Y_MARK)
    except ValueError as exc:
        gui.update_status(process='ERROR')
        gui.log_step(str(exc))
        return False

    color = resolve_color(command.get('colour'))
    strokes = get_draw_strokes(command, cx=0.0, cy=Y_MARK)
    gui.update_status(servo=None)
    gui.log_step(f"Running mode={command.get('mode')} shape={getattr(draw_shape, 'kind', '?')} "
                f"colour={command.get('colour')}")
    try:
        run_cycle(env, arms, box, ink_pool, dt=dt, shape=draw_shape, color=color,
                strokes=strokes, controllers=controllers)
        check_continue(env, dt)
    except gui.CycleAborted:
        # Finish resetting arms and scene in the main loop, after this task exits.
        gui.acknowledge_abort()
        gui.log_step('Cycle aborted (E-STOP + RESET PROCESS)')
        return False
    gui.update_status(process='COMPLETE')
    gui.log_step('Cycle complete')
    return True

# Puts the box back in the storage stand, removes the stamps and gives all ink marks back to the pool
def reset_scene(box, ink_pool):
    box.reset(*STORAGE_XY, STAND_TOP, pool=ink_pool)
    ink_pool.reset()

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
    ok &= arm2.check_look(0.0, Y_MARK, BOX_TOP, CAM_HEIGHT)
    print('  layout OK\n' if ok else '  LAYOUT PROBLEM\n')
    return ok

if __name__ == "__main__":
    check_layout()

    env = swift.Swift()
    env.launch(realtime=True)

    scene = build_workcell(env)
    arms = make_arms()
    for arm in arms:
        arm.add_to_env(env)

    # Wrist camera model on the Staubli (arms[1]); play() keeps its pose in sync with the arm
    camera_model = make_camera_model()
    env.add(camera_model)
    arms[1].cam_model = camera_model
    camera_model.T = arms[1].cam_pose().A
    
    ink_pool = MarkPool(env, INK_POOL_SIZE, lambda: Sphere(radius=INK_RADIUS, color=INK))

    env.set_camera_pose([2.9, -2.6, 2.1], [0, 0, 0.8])
    env.step(0.02)

    # One safety controller per arm, in the same order as the arms (a list, not a set).
    # The same objects are given to the GUI, so the E-STOP button acts on them.
    controllers = [RobotSafetyController(robot_name=name) for name in gui.ARM_NAMES]

    gui.start_upload_server()           # local page used to import images
    widgets = gui.build_controls(env, arms, controllers, scene)
    physical_estop.start()
    gui.update_status(process='READY')
    gui.refresh_status_label(widgets)
    print('Swift page open with GUI widgets embedded. Waiting for commands...')

    box = MarkedBox(env, *STORAGE_XY, STAND_TOP)        # one box, reused by every cycle
    scene_dirty = False                 # True once a cycle has used the box / the ink marks
    running = True
    while running:
        gui.handle_upload_request()     # imports images received by the upload page, if any
        if gui.consume_reset_request():     # RESET PROCESS: box back in storage, marks cleared
            reset_scene(box, ink_pool)
            gui.finish_reset_process()
            scene_dirty = False
            gui.log_step('Box back in storage, marks cleared')
        command = gui.get_pending_command()
        if command is not None:
            if command.get('mode') == 'stop':
                running = False
                break
            if scene_dirty:             # RUN without RESET PROCESS: start from a clean scene anyway
                reset_scene(box, ink_pool)
            scene_dirty = True
            run_gui_cycle(env, arms, box, ink_pool, command, controllers=controllers)

        gui.refresh_status_label(widgets)
        env.step(0.02)

    gui.stop_upload_server()
    env.hold()
    time.sleep(3)