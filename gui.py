# Including libraries
import http.server
import math
import os
import queue
import shutil
import tempfile
import threading
import time
import urllib.parse
import webbrowser

import numpy as np
import swift
from spatialmath import SE3

from safety import CollisionSafety, SingularitySafety, RobotSafetyController
from workcell import CRATE_CENTRE, CRATE_SIZE, MarkPool

status = {
    "process": "READY",      # READY | RUNNING | COMPLETE | STOPPED | ERROR | E-STOP
    "mode": None,             # "random" | "basic" | "image"
    "shape": None,             # "circle" | "square" | "triangle" | "complex"
    "colour": None,
    "image_name": None,
    "image_ready": False,
    "servo": None,
    "log": [],
}

image_models = {}
image_strokes = {}          # filename -> list of strokes (outer outline first), see sensing.strokes_from_jpeg

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "dogoodbot_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Image formats accepted by the upload page (keep in sync with sensing.SUPPORTED_EXTENSIONS)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# The upload page is a tiny HTTP server that only listens on this machine
UPLOAD_HOST = '127.0.0.1'
UPLOAD_PORT = 8765                      # the following ports are tried if this one is busy
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Keep the accepted extensions and the 10 MB limit in sync with IMAGE_EXTENSIONS and MAX_UPLOAD_BYTES
UPLOAD_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DoGoodBot - Upload Image</title>
<style>
    body { font-family: sans-serif; max-width: 480px; margin: 40px auto; padding: 0 16px; }
    button { margin-left: 8px; padding: 4px 16px; }
    #msg { margin-top: 16px; }
</style>
</head>
<body>
<h2>DoGoodBot: Upload Image</h2>
<input type="file" id="file" accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp">
<button id="send">Upload</button>
<p id="msg">Choose an image (max 10 MB), then press Upload.</p>
<script>
document.getElementById('send').onclick = async () => {
    const file = document.getElementById('file').files[0];
    const msg = document.getElementById('msg');
    if (!file) { msg.textContent = 'Choose an image first.'; return; }
    if (file.size > 10 * 1024 * 1024) { msg.textContent = 'File too large (max 10 MB).'; return; }
    msg.textContent = 'Uploading...';
    try {
    const reply = await fetch('/upload?name=' + encodeURIComponent(file.name), {
        method: 'POST',
        headers: {'X-Requested-With': 'dogoodbot'},
        body: file
    });
    msg.textContent = await reply.text();
    } catch (err) {
    msg.textContent = 'Upload failed: ' + err;
    }
};
</script>
</body>
</html>
"""

# Paint colour is picked with three RGB sliders (0-255), starting from this value.
DEFAULT_RGB = (200, 40, 40)
_rgb = list(DEFAULT_RGB)

# Selections made via the on-page Select widgets.
_selected_mode = 'random'
_selected_shape = None
_selected_colour = DEFAULT_RGB

_pending_command = None

# Paths of images saved by the upload page; the main loop imports them (see handle_upload_request)
_uploaded_files = queue.Queue()
_upload_server = None
_upload_url = None

# Arms controlled by the joint sliders (set by build_controls), the six slider widgets, and the index of the arm the sliders currently drive
_arms = []
_joint_sliders = []
_selected_arm = 0

# Names shown on the arm Select, in the same order as make_arms()
ARM_NAMES = ['Kawasaki', 'Staubli', 'Nachi']

# Number of joint sliders (every arm has 6 joints)
N_JOINTS = 6

# Safety monitors and one RobotSafetyController per arm (same order as ARM_NAMES)
_collision = CollisionSafety()
_singularity = SingularitySafety()
_controllers = []
_widgets = {}
_crate = None
_crate_active = False
_scene = {}
_curtain_broken = False
_last_person_time = None

# Latest result per arm index: (collision result, singularity result, speed set by operator, speed allowed)
_last_safety = {}

# True from RUN until the cycle ends (COMPLETE / ERROR) or is aborted; RESET PROCESS after an E-STOP sets _abort_requested
_cycle_running = False
_abort_requested = False

# Set by RESET PROCESS; main.py reads it (consume_reset_request) and puts the box back in storage and clears the marks
_reset_requested = False

# Timing (seconds) for label refresh, joint slider sync during RMRC jog
LABEL_PERIOD = 0.2
JOG_SYNC_PERIOD = 0.3
_last_label_time = 0.0
_last_jog_time = time.time()
_last_jog_sync = 0.0

# RMRC jog: desired end-effector velocity [vx, vy, vz, wx, wy, wz] in m/s and rad/s, in the base frame of the selected arm.
# One slider (percent of the axis maximum) drives the axis chosen with the RMRC Axis Select
_rmrc_v = [0.0] * 6
_rmrc_axis = 0
_rmrc_slider = None
_speed_slider = None
RMRC_LABELS = ['Vx (mm/s)', 'Vy (mm/s)', 'Vz (mm/s)', 'Wx (deg/s)', 'Wy (deg/s)', 'Wz (deg/s)']
RMRC_LIMITS = [100, 100, 100, 30, 30, 30]

class CycleAborted(Exception):
    """
    Raised by play() in main.py when RESET PROCESS aborts a running cycle.
    """

def update_status(**fields):
    global _cycle_running
    status.update(fields)
    if fields.get('process') in ('COMPLETE', 'ERROR'):
        _cycle_running = False

def log_step(message, max_lines=12):
    print(f'[gui] {message}')
    status["log"].append(message)
    status["log"] = status["log"][-max_lines:]

def get_status_snapshot():
    return dict(status)

def select_mode(mode):
    global _selected_mode
    _selected_mode = mode
    log_step(f'Mode set to {mode}')

def select_shape(shape):
    global _selected_shape
    changed = shape != _selected_shape
    _selected_shape = shape
    if changed and shape is not None:
        log_step(f'Shape: {shape}')

def select_colour(colour):
    global _selected_colour
    _selected_colour = colour
    log_step(f'Colour set to {colour}')

def select_rgb(channel, value):
    global _selected_colour
    _rgb[channel] = int(value)
    _selected_colour = tuple(_rgb)

# SAFETY: collision, singularity, speed

def get_controllers():
    return _controllers

def _estopped():
    return any(c.is_estopped() for c in _controllers)

def _arm_index(arm):
    for i, a in enumerate(_arms):
        if a is arm:
            return i
    return None

def check_crate(arm, q=None):
    if not _crate_active or _crate is None or _reset_requested:
        return None
    result = _collision.box_obstacle(arm, np.asarray(_crate.T)[:3, 3], CRATE_SIZE, q)
    if not result['safe'] and not _estopped():
        request_stop('Crate collision risk; E-STOP active')
    return result

# Collision (self + every other arm) and singularity result of one arm at joint configuration q (None = current pose)
def evaluate_safety(arm, q=None):
    results = [_collision.self_collision(arm, q)]
    crate = check_crate(arm, q)
    if crate is not None:
        results.append(crate)
    for other in _arms:
        if other is arm:
            continue
        results.append(_collision.robot_robot(arm, other, q1=q))
    # Worst result wins, but the reported distance is the smallest one found
    collision = dict(CollisionSafety.merge(*results), distance=min(r["distance"] for r in results))
    singularity = _singularity.check(arm, q)
    return collision, singularity

def _record_safety(index, collision, singularity, controller, scale):
    if index is not None:
        _last_safety[index] = (collision, singularity, controller.manual_scale, scale)

# Final speed scale (0..1) for an arm at q: E-STOP, collision and severe singularity give 0, warnings and
# nearby obstacles only slow the robot down. Called by play() in main.py for every trajectory point.
# During a cycle, a collision or severe singularity latches the E-STOP (so the Nucleo is told as well)
def safety_scale(arm, q=None, controller=None):
    index = _arm_index(arm)
    if controller is None:
        controller = _controllers[index]
    collision, singularity = evaluate_safety(arm, q)
    # A serial RESET may have aborted this task while evaluate_safety was running.
    if _cycle_running and not _reset_requested and not _estopped():
        if not collision["safe"] or singularity["level"] == "STOP":
            reason = collision["reason"] if not collision["safe"] else singularity["reason"]
            log_step(f'Safety stop: {reason}')
            request_stop()
    scale = controller.get_velocity_scale(collision, singularity)
    _record_safety(index, collision, singularity, controller, scale)
    return scale

# Speed limit in percent of the selected robot; the allowed speed is reduced automatically when needed
def select_speed(percent):
    if _controllers:
        _controllers[_selected_arm].set_manual_scale(float(percent) / 100.0)

# Puts the speed slider in line with the selected robot
def _sync_speed_slider():
    if _speed_slider is not None and _controllers:
        percent = int(round(_controllers[_selected_arm].manual_scale * 100))
        _speed_slider.value = max(5, min(100, percent))

# Chooses which velocity component the RMRC slider drives (0-2 linear in mm/s, 3-5 angular in deg/s)
def select_rmrc_axis(index):
    global _rmrc_axis
    _rmrc_axis = index
    _zero_rmrc()

# Velocity of the chosen axis as percent (-100..100) of its maximum (RMRC_LIMITS)
def select_rmrc(percent):
    unit = 1e-3 if _rmrc_axis < 3 else math.pi / 180.0
    _rmrc_v[:] = [0.0] * 6
    _rmrc_v[_rmrc_axis] = float(percent) / 100.0 * RMRC_LIMITS[_rmrc_axis] * unit

def _zero_rmrc():
    _rmrc_v[:] = [0.0] * 6
    if _rmrc_slider is not None:
        _rmrc_slider.value = 0

def abort_requested():
    return _abort_requested

# Called by main.py after it caught CycleAborted
def acknowledge_abort():
    global _abort_requested, _cycle_running
    _abort_requested = False
    _cycle_running = False
    update_status(process='READY')

# True until main.py has reset the box/marks and called finish_reset_process.
def consume_reset_request():
    return _reset_requested  # Cleared after the main loop finishes the reset.

# True if moving to q_new does not create a collision (a move that leaves an existing unsafe pose is allowed)
def _joint_move_allowed(arm, q_new):
    current, _ = evaluate_safety(arm)
    new, _ = evaluate_safety(arm, q_new)
    if new["safe"]:
        return True
    if current["safe"] or new.get("distance", 0.0) < current.get("distance", 0.0):
        log_step(f"Joint move blocked: {new['reason']} ({new.get('distance', 0.0):.3f} m)")
        return False
    return True

# Moves the selected arm with RMRC (called every loop by refresh_status_label); ignored during a cycle or E-STOP
def step_rmrc_jog():
    global _last_jog_time, _last_jog_sync
    now = time.time()
    dt = min(now - _last_jog_time, 0.1)
    _last_jog_time = now
    if not _arms or not _controllers or not any(_rmrc_v) or _cycle_running or _reset_requested:
        return
    index = _selected_arm
    arm = _arms[index]
    controller = _controllers[index]
    collision, singularity = evaluate_safety(arm)
    scale = controller.get_velocity_scale(collision, singularity)
    _record_safety(index, collision, singularity, controller, scale)
    qdot = controller.rmrc_joint_velocity(arm, _rmrc_v, collision, singularity)
    if not np.any(qdot):
        return
    qlim = arm.robot.qlim
    q_new = np.clip(arm.robot.q + qdot * dt, qlim[0], qlim[1])
    crate = check_crate(arm, q_new)
    if crate is not None and not crate['safe']:
        return
    arm.robot.q = q_new
    if now - _last_jog_sync > JOG_SYNC_PERIOD:
        sync_joint_sliders()
        _last_jog_sync = now

# Limits of one joint in whole degrees, rounded outwards so the slider always covers qlim
def _joint_limits_deg(arm, joint):
    qlim = arm.robot.qlim
    return (int(math.floor(math.degrees(qlim[0][joint]))),
            int(math.ceil(math.degrees(qlim[1][joint]))))

# Moves one joint of one arm (value in degrees, clamped to the joint limits); ignored while a cycle is running because play() owns arm.robot.q then,
# and blocked during E-STOP or when the move would create a collision
def select_joint(arm, joint, value):
    if _cycle_running or _reset_requested:
        return
    if _estopped():
        log_step('E-STOP active: press RESET PROCESS first')
        sync_joint_sliders()
        return
    qlim = arm.robot.qlim
    q = arm.robot.q.copy()
    q[joint] = min(max(math.radians(float(value)), qlim[0][joint]), qlim[1][joint])
    if not _joint_move_allowed(arm, q) or _estopped():
        sync_joint_sliders()
        return
    arm.robot.q = q

# Puts the six joint sliders (limits and value) in line with the arm they currently control
def sync_joint_sliders():
    if not _arms:
        return
    arm = _arms[_selected_arm]
    for joint, slider in enumerate(_joint_sliders):
        lo, hi = _joint_limits_deg(arm, joint)
        val = int(round(math.degrees(arm.robot.q[joint])))
        slider.min = lo
        slider.max = hi
        slider.value = max(lo, min(hi, val))

# Chooses which arm the joint sliders, the speed slider and the RMRC slider control
def select_arm(index):
    global _selected_arm
    if not 0 <= index < len(_arms):
        return
    changed = index != _selected_arm
    _selected_arm = index
    _zero_rmrc()
    _sync_speed_slider()
    sync_joint_sliders()
    if 'adjust_label' in _widgets:
        _widgets['adjust_label'].desc = _adjust_text()
    if changed:
        log_step(f'Robot: {ARM_NAMES[index]}')

# RESET ARMS: only sends every arm back to its home pose (use it after the joints were moved by hand).
# Refused while a cycle is running and during an E-STOP (RESET PROCESS clears the E-STOP)
def reset_arms():
    if _cycle_running or _reset_requested:
        log_step('Cannot reset arms while a cycle is running (press E-STOP, then RESET PROCESS)')
        return False
    if _estopped():
        log_step('E-STOP active: press RESET PROCESS')
        return False
    for arm in _arms:
        arm.robot.q = arm.HOME.copy()
    _zero_rmrc()
    sync_joint_sliders()
    log_step('All arms reset to home')
    return True

# RESET PROCESS requests a reset; main.py applies the poses after the old task exits.
# The physical bridge calls this same callback from its serial thread.
def reset_process():
    global _abort_requested, _reset_requested, _pending_command
    person = _scene.get('person')
    if person is not None and person.reset_blocked():
        log_step('Reset blocked: move person outside first')
        return False
    if _cycle_running and not _estopped():
        log_step('Cannot reset while a cycle is running (press E-STOP first)')
        return False
    _reset_requested = True
    _abort_requested = _cycle_running
    _pending_command = None
    _zero_rmrc()
    for controller in _controllers:
        controller.reset_estop()
    update_status(process='RESETTING', servo=None)
    return True

def finish_reset_process():
    global _reset_requested, _abort_requested, _cycle_running, _curtain_broken, _crate_active
    person = _scene.get('person')
    if person is not None and person.reset_blocked():
        request_stop('Reset cancelled: person entering/inside')
        _abort_requested = _cycle_running = _reset_requested = False
        return
    for arm in _arms:
        arm.robot.q = arm.HOME.copy()
    for controller in _controllers:
        controller.set_manual_scale(1.0)
    _zero_rmrc()
    sync_joint_sliders()
    _sync_speed_slider()
    # A new E-stop received after reset_process must remain active.
    if not _estopped():
        _crate_active = False
        if _crate is not None:
            _crate.T = MarkPool.PARKED.A
        _curtain_broken = False
        if 'person_select' in _widgets:
            _widgets['person_select'].value = 0
    _last_safety.clear()
    _abort_requested = _cycle_running = _reset_requested = False
    update_status(process='E-STOP' if _estopped() else 'READY', servo=None)
    log_step('Reset complete; E-STOP active' if _estopped() else 'Reset complete; READY')

def add_crate():
    global _crate_active
    if _crate is not None:
        _crate.T = SE3(*CRATE_CENTRE).A
        _crate_active = True
        log_step('Crate added; collision monitoring active')

def set_light_curtain(broken):
    global _curtain_broken
    changed = bool(broken) != _curtain_broken
    _curtain_broken = bool(broken)
    if _curtain_broken and (changed or not _estopped()):
        request_stop('Curtain interrupted; E-STOP active')
    elif changed:
        log_step('Curtain clear; reset required' if _estopped() else 'Curtain clear')

def move_person(enter):
    person = _scene.get('person')
    if person is not None:
        person.request_enter(bool(enter))
        log_step('Person entering workplace' if enter else 'Person leaving workplace')

def check_light_curtain():
    global _last_person_time
    now = time.monotonic()
    dt = 0. if _last_person_time is None else now - _last_person_time
    _last_person_time = now
    person = _scene.get('person')
    if person is not None:
        person.step(dt)
        set_light_curtain(person.blocks_curtain())

def process_image_file(path, target_size=0.035):
    filename = os.path.basename(path)
    try:
        import sensing
        model = sensing.shape_model_from_jpeg(path, target_size=target_size)
        image_models[filename] = model
        try:
            image_strokes[filename] = sensing.strokes_from_jpeg(path, target_size=target_size)
            log_step(f'{filename}: {len(image_strokes[filename])} stroke(s) extracted')
        except Exception as stroke_exc:
            image_strokes.pop(filename, None)       # falls back to the single outline
            log_step(f'{filename}: strokes failed ({stroke_exc}), using single outline')
        update_status(image_name=filename, image_ready=True)
        log_step(f'{filename}: contour extracted, ready to run')
        return True
    except Exception as exc:
        update_status(image_name=filename, image_ready=False)
        log_step(f'{filename}: processing failed - {exc}')
        return False

# Copies the chosen image into UPLOAD_DIR (skipped if it is already there), then runs the normal processing pipeline
def import_image(path):
    filename = os.path.basename(path)
    dest = os.path.join(UPLOAD_DIR, filename)
    try:
        if os.path.abspath(path) != os.path.abspath(dest):
            shutil.copy2(path, dest)
    except OSError as exc:
        update_status(image_name=filename, image_ready=False)
        log_step(f'{filename}: copy failed - {exc}')
        return False
    return process_image_file(dest)

# Serves the upload page and receives the image bytes. It only saves the file and queues its path;
# the image itself is processed by the main loop, not by this server thread.
class UploadHandler(http.server.BaseHTTPRequestHandler):
    timeout = 15

    def _reply(self, code, body, content_type='text/plain; charset=utf-8'):
        data = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # Returns the request body, or None if it is larger than MAX_UPLOAD_BYTES
    def _read_body(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
        except ValueError:
            length = 0
        if length > MAX_UPLOAD_BYTES:
            # drain the body so the browser can still read the reply
            remaining = min(length, 10 * MAX_UPLOAD_BYTES)
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            return None
        return self.rfile.read(length) if length > 0 else b''

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path == '/':
            self._reply(200, UPLOAD_PAGE, 'text/html; charset=utf-8')
        else:
            self._reply(404, 'Not found')

    def do_POST(self):
        data = self._read_body()
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != '/upload':
            return self._reply(404, 'Not found')
        # A custom header makes other websites unable to post here (the browser blocks it cross-origin)
        if self.headers.get('X-Requested-With') != 'dogoodbot':
            return self._reply(403, 'Forbidden')
        if data is None:
            return self._reply(413, f'File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)')

        raw_name = urllib.parse.parse_qs(parsed.query).get('name', [''])[0]
        name = os.path.basename(raw_name.replace('\\', '/'))
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            return self._reply(400, f'Unsupported file type, use one of {IMAGE_EXTENSIONS}')
        if not data:
            return self._reply(400, 'Empty file')

        dest = os.path.join(UPLOAD_DIR, name)
        tmp = dest + '.part'
        try:
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, dest)       # the main loop never sees a half-written file
        except OSError as exc:
            return self._reply(500, f'Could not save {name}: {exc}')
        _uploaded_files.put(dest)
        log_step(f'{name}: received by the upload page')
        self._reply(200, f'Received {name}. Check the Swift page for the result.')

    def log_message(self, *args):
        pass                            # keep the terminal quiet

# Starts the upload page in a background thread and returns its URL (None if no port is free)
def start_upload_server(port=UPLOAD_PORT, tries=10):
    global _upload_server, _upload_url
    if _upload_server is not None:
        return _upload_url
    ports = [0] if port == 0 else range(port, port + tries)
    for p in ports:
        try:
            server = http.server.ThreadingHTTPServer((UPLOAD_HOST, p), UploadHandler)
        except OSError:
            continue
        _upload_server = server
        _upload_url = f'http://{UPLOAD_HOST}:{server.server_address[1]}'
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log_step(f'Upload page ready at {_upload_url}')
        return _upload_url
    log_step('Upload page unavailable: no free port')
    return None

def stop_upload_server():
    global _upload_server, _upload_url
    if _upload_server is None:
        return
    _upload_server.shutdown()
    _upload_server.server_close()
    _upload_server = None
    _upload_url = None

# Opens the upload page in the default browser; the link is also logged in case no browser can be opened
def open_upload_page():
    if _upload_url is None:
        log_step('Upload page is not running')
        return False
    log_step(f'Upload page: {_upload_url}')
    try:
        opened = webbrowser.open(_upload_url)
    except Exception as exc:
        log_step(f'Could not open the browser: {exc}')
        return False
    if not opened:
        log_step('Could not open the browser, open the link above manually')
    return bool(opened)

# Called every loop iteration by main.py: imports the images received by the upload page, if any
def handle_upload_request():
    imported = False
    while True:
        try:
            path = _uploaded_files.get_nowait()
        except queue.Empty:
            break
        imported = import_image(path) or imported
    return imported

def request_run():
    global _pending_command, _cycle_running

    if _cycle_running or _reset_requested:
        log_step('Cannot start: a task is running or reset is pending')
        return False

    if _estopped():
        log_step('START blocked; reset required')
        return False
    if _selected_mode is None:
        log_step('Cannot run: no mode selected')
        return False
    if _selected_mode == 'basic' and _selected_shape is None:
        log_step('Cannot run: no shape selected for Basic Shape mode')
        return False
    if _selected_mode == 'image' and not status.get('image_ready'):
        log_step('Cannot run: process an image first')
        return False

    _pending_command = {'mode': _selected_mode, 'shape': _selected_shape,
                        'colour': _selected_colour}
    _cycle_running = True
    update_status(process='RUNNING', mode=_selected_mode,
                shape=_selected_shape, colour=_selected_colour)
    log_step(f"Queued run: mode={_selected_mode}, shape={_selected_shape}, "
            f"colour={_selected_colour}")
    return True

# STOP button = E-STOP: every robot stops immediately (play() in main.py holds position) until RESET PROCESS is pressed
def request_stop(reason='E-STOP active; reset required'):
    global _pending_command
    _pending_command = None
    for controller in _controllers:
        controller.trigger_estop()
    update_status(process='E-STOP')
    log_step(reason)

def get_pending_command():
    global _pending_command
    command, _pending_command = _pending_command, None
    return command

def _selected_index(event):
    return int(getattr(event, 'number', event))

def build_controls(env, arms=None, controllers=None, scene=None):
    global _selected_arm, _rmrc_slider, _speed_slider, _crate, _scene, _last_person_time
    _scene = scene or {}
    _last_person_time = time.monotonic()
    _crate = _scene.get('crate')
    widgets = {}
    env.add(swift.Label(desc='Task Execution'))

    def on_mode_changed(event):
        select_mode(['random', 'basic', 'image'][_selected_index(event)])

    widgets['mode_select'] = swift.Select(
        cb=on_mode_changed,
        desc='Process Mode',
        options=['Random Shape', 'Basic Shape', 'Import Image'], value=0)
    env.add(widgets['mode_select'])

    def on_shape_changed(event):
        select_shape([None, 'circle', 'square', 'triangle'][_selected_index(event)])

    widgets['shape_select'] = swift.Select(
        cb=on_shape_changed,
        desc='Basic Shape',
        options=['Choose shape', 'Circle', 'Square', 'Triangle'], value=0)
    env.add(widgets['shape_select'])

    # Swift has no file-upload widget: the button opens a local upload page in the browser
    widgets['image_label'] = swift.Label(desc=_image_text())
    env.add(widgets['image_label'])

    widgets['upload_button'] = swift.Button(cb=lambda *_: open_upload_page(), desc='Upload Image')
    env.add(widgets['upload_button'])

    # One slider per RGB channel controls the paint colour.
    for i, name in enumerate(['R', 'G', 'B']):
        widgets[f'{name.lower()}_slider'] = swift.Slider(
            cb=lambda value, i=i: select_rgb(i, value),
            min=0, max=255, step=1,
            value=_rgb[i],
            desc=f'Paint {name}')
        env.add(widgets[f'{name.lower()}_slider'])

    widgets['run_button'] = swift.Button(cb=lambda *_: request_run(), desc='START')
    env.add(widgets['run_button'])

    env.add(swift.Label(desc='Emergency Stop'))

    widgets['stop_button'] = swift.Button(cb=lambda *_: request_stop(), desc='E-STOP')
    env.add(widgets['stop_button'])

    widgets['reset_process_button'] = swift.Button(cb=lambda *_: reset_process(), desc='RESET PROCESS')
    env.add(widgets['reset_process_button'])

    env.add(swift.Label(desc='Safety Tests'))
    widgets['person_select'] = swift.Select(
        cb=lambda event: move_person(_selected_index(event) == 1),
        desc='Person', options=['Outside / leave workplace', 'Enter workplace'], value=0)
    env.add(widgets['person_select'])
    widgets['curtain_label'] = swift.Label(desc='Light curtain: clear | Person outside')
    env.add(widgets['curtain_label'])
    widgets['crate_button'] = swift.Button(cb=lambda *_: add_crate(), desc='Add crate')
    env.add(widgets['crate_button'])
    widgets['safety_label'] = swift.Label(desc=_safety_text())
    env.add(widgets['safety_label'])

    env.add(swift.Label(desc='Manual Teach Pendant'))
    # A Select picks the arm, six joint sliders (degrees) drive the picked arm within its joint limits, plus a reset button
    if arms:
        _arms[:] = list(arms)
        _selected_arm = 0
        _joint_sliders.clear()

        # One safety controller per arm (the ones from main.py if given, in the same order as the arms)
        _controllers[:] = list(controllers) if controllers else [
            RobotSafetyController(robot_name=name) for name in ARM_NAMES[:len(_arms)]]

        def on_arm_changed(event):
            select_arm(_selected_index(event))

        widgets['arm_select'] = swift.Select(
            cb=on_arm_changed,
            desc='Controlled Robot',
            options=list(ARM_NAMES[:len(_arms)]), value=0)
        env.add(widgets['arm_select'])

        arm = _arms[_selected_arm]
        for j in range(N_JOINTS):
            lo, hi = _joint_limits_deg(arm, j)
            val = int(round(math.degrees(arm.robot.q[j])))
            slider = swift.Slider(
                cb=lambda value, j=j: select_joint(_arms[_selected_arm], j, value),
                min=lo, max=hi, step=1,
                value=max(lo, min(hi, val)),
                desc=f'Joint J{j + 1} (deg)')
            env.add(slider)
            widgets[f'j{j + 1}_slider'] = slider
            _joint_sliders.append(slider)

        widgets['reset_button'] = swift.Button(cb=lambda *_: reset_arms(), desc='RESET ARMS')
        env.add(widgets['reset_button'])

        # Shows which robot the speed slider and the RMRC controls act on
        widgets['adjust_label'] = swift.Label(desc=_adjust_text())
        env.add(widgets['adjust_label'])

        # Speed limit in percent of the robot chosen with the Select above, reduced automatically near collisions or singularities
        _speed_slider = swift.Slider(
            cb=lambda value: select_speed(value),
            min=5, max=100, step=5,
            value=100,
            desc='Speed limit (%)')
        env.add(_speed_slider)
        widgets['speed_slider'] = _speed_slider

        # RMRC: a Select picks the velocity component, one slider (percent of that component's maximum) moves the chosen robot
        widgets['rmrc_axis_select'] = swift.Select(
            cb=lambda event: select_rmrc_axis(_selected_index(event)),
            desc='RMRC Axis',
            options=list(RMRC_LABELS), value=0)
        env.add(widgets['rmrc_axis_select'])

        _rmrc_slider = swift.Slider(
            cb=lambda value: select_rmrc(value),
            min=-100, max=100, step=1,
            value=0,
            desc='RMRC velocity (% of axis max)')
        env.add(_rmrc_slider)
        widgets['rmrc_slider'] = _rmrc_slider

    widgets['status_label'] = swift.Label(desc=_status_text())
    env.add(widgets['status_label'])

    env.add(swift.Label(desc='Events'))
    for i in range(3):
        widgets[f'event_{i}'] = swift.Label(desc='')
        env.add(widgets[f'event_{i}'])

    _widgets.clear()
    _widgets.update(widgets)
    return widgets


def _status_text():
    snap = get_status_snapshot()
    text = f"Process: {snap['process']} | Mode: {snap['mode'] or '-'} | " \
        f"Shape: {snap['shape'] or '-'} | Colour: {snap['colour'] or '-'}"
    if snap.get('image_name'):
        text += f" | Image: {snap['image_name']} ({'ready' if snap['image_ready'] else 'processing'})"
    if snap.get('servo'):
        text += f" | Servo: {snap['servo']}"
    return text

def _image_text():
    snap = get_status_snapshot()
    text = 'Image: No image selected'
    if snap.get('image_name'):
        text = f"Image: {snap['image_name']} ({'ready' if snap['image_ready'] else 'processing'})"
        n = len(image_strokes.get(snap['image_name'], ()))
        if n > 1:
            text += f' | {n} stroke(s)'
    if _upload_url:
        text += f' | Upload page: {_upload_url}'
    return text

def _adjust_text():
    if not _arms:
        return 'Adjusting: -'
    return f'Adjusting: {ARM_NAMES[_selected_arm]}'

def _safety_text():
    if not _arms or not _controllers:
        return 'Safety: -'
    name = ARM_NAMES[_selected_arm]
    estop = ' | E-STOP ACTIVE' if _estopped() else ''
    entry = _last_safety.get(_selected_arm)
    if entry is None:
        return f'Safety ({name}): -{estop}'
    collision, singularity, set_scale, allowed = entry
    distance = collision.get('distance', float('nan'))
    text = (f"Safety ({name}) | Collision: {'OK' if collision['safe'] else 'RISK'} ({distance:.2f} m)"
            f" | Singularity: {singularity['level']} (sigma_min {singularity['sigma_min']:.3f})"
            f" | Speed: set {set_scale * 100:.0f}% -> allowed {allowed * 100:.0f}%")
    if allowed < set_scale - 1e-9:
        text += ' (reduced)'
    return text + estop

def refresh_status_label(widgets=None):
    global _last_label_time
    if widgets is None:
        widgets = _widgets

    check_light_curtain()

    # Check all current poses, including stationary arms, before manual motion.
    for arm in _arms:
        check_crate(arm)
    step_rmrc_jog()
    for arm in _arms:
        cam_model = getattr(arm, 'cam_model', None)
        if cam_model is not None:
            cam_model.T = arm.cam_pose().A

    if _scene:
        _scene['light_curtain'].show(_curtain_broken)
        red = _estopped()
        states = dict(red=red, green=_cycle_running and not red and not _reset_requested,
                    amber=(not _cycle_running or _reset_requested) and not red)
        for name, light in _scene['tower'].items():
            light.show(states[name])

    now = time.time()
    if now - _last_label_time < LABEL_PERIOD:
        return
    _last_label_time = now

    # While no cycle runs the safety of the selected arm is checked here (during a cycle play() does it)
    if _arms and _controllers and not _cycle_running:
        safety_scale(_arms[_selected_arm])

    if 'status_label' in widgets:
        widgets['status_label'].desc = _status_text()
    if 'image_label' in widgets:
        widgets['image_label'].desc = _image_text()
    if 'safety_label' in widgets:
        widgets['safety_label'].desc = _safety_text()
    if 'curtain_label' in widgets:
        person = _scene.get('person')
        occupied = person is not None and person.reset_blocked()
        widgets['curtain_label'].desc = (
            'Light curtain: ' + ('INTERRUPTED' if _curtain_broken else 'clear') +
            (' | Person entering/inside; reset blocked' if occupied else ' | Person outside') +
            (' | E-STOP latched' if _estopped() else ''))
    if 'adjust_label' in widgets:
        widgets['adjust_label'].desc = _adjust_text()
    for i, line in enumerate(([''] * 3 + status['log'])[-3:]):
        if f'event_{i}' in widgets:
            widgets[f'event_{i}'].desc = line if len(line) <= 90 else line[:87] + '...'