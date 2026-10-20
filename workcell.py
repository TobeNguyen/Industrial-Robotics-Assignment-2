# Including libraries
import os
import sys
import tempfile

import numpy as np
import trimesh
from spatialmath import SE3
from spatialmath.base import trnorm
from spatialgeometry import Axes, Cuboid, Cylinder, Mesh, Sphere

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(current_path, 'tasks'))

from kawasaki_rs007n_task import KawasakiPickPlace      # Student 1
from staubli_tx2_60_task import StaubliShapeDraw        # Student 2
from nachi_mz07_task import NachiShapeFill              # Student 3

# Conveyor
# Third-party CAD, Apache-2.0, see environment/ATTRIBUTION.md. Measured from
# the file, do NOT scale it: 0.463 wide x 1.200 long x 0.743 tall, belt surface
# at z = 0.741, origin on the floor and centred in x and y.
CONVEYOR_DAE = os.path.join(current_path, 'environment', 'conveyor_belt.dae')
SECTION_LENGTH = 1.200
SECTIONS = 2                            # butted end to end -> 2.400 m of belt
BELT_Z = 0.741
BELT_WIDTH = 0.463
BELT_LENGTH = SECTIONS * SECTION_LENGTH

# Robot mounting
# The belt sits at 0.741 m, which a floor-mounted TX2-60 cannot reach, so all
# three arms stand on pedestals. 0.400 m puts the belt 0.341 m above each base.
PEDESTAL_H = 0.400
PEDESTAL_W = 0.300
STANDOFF = 0.460                        # robot base to belt centreline
STAGGER = 0.750                         # spacing between arms along the belt

# Parts
BOX = [0.120, 0.120, 0.090]
BOX_TOP = BELT_Z + BOX[2]               # lid height once the box is on the belt
BOX_HALF = BOX[0] / 2
STORAGE_XY = (-0.880, -1.000)
FINISH_XY = (-0.880, -0.500)
STAND_TOP = BELT_Z                      # storage and finish stands match the belt

# Stations
Y_LOAD = -STAGGER                       # Kawasaki station
Y_MARK = 0.0                            # Staubli station
Y_PAINT = STAGGER                       # Nachi station

# Shape drawn per cycle
# Half-size used for whichever shape ShapeSpec picks: circle radius, square
# half-side, or triangle half-height. Kept under BOX_HALF so the shape stays
# on the lid whichever kind is drawn.
SHAPE_SIZE = 0.035

# Safety Equipment
FENCE_X, FENCE_Y, FENCE_H = 1.400, 1.750, 1.800

# Layout values from the user's workcell_test.py (simulation parameters).
REACH_KAWASAKI = 0.355 + 0.375 + 0.078
REACH_NACHI = 0.050 + 0.330 + np.hypot(0.045, 0.340) + 0.083
HAZARD_X = -STANDOFF - max(REACH_KAWASAKI, REACH_NACHI) - 0.040
SAFETY_DISTANCE = 0.408  # Original demo: T=0.140 s, d=30 mm -> 408 mm.
CURTAIN_X = HAZARD_X - SAFETY_DISTANCE
OPEN_HALF = 1.40
CURTAIN_BEAM_Z = np.linspace(0.25, FENCE_H - 0.25, 7)
CURTAIN_BEAM_HALF = 0.006
GROUND_DEPTH = 3.0
CRATE_CENTRE = (-0.36, -1.25, 0.475)
CRATE_SIZE = (0.20, 0.20, 0.95)

STEEL = (0.62, 0.65, 0.68, 1)
DARK = (0.42, 0.44, 0.46, 1)
YELLOW = (0.85, 0.72, 0.15, 1)
CARDBOARD = (0.76, 0.60, 0.38, 1)
PAINTED = (0.20, 0.45, 0.78, 1)         # default stamp colour, used when the
                                        # operator hasn't picked one (or for
                                        # callers/tests that don't pass color)
INK = (0.10, 0.10, 0.12, 1)

# How often an ink mark is laid down while the draw tool is on the lid. The
# path has one point per degree, far finer than the ink needs to look
# continuous.
INK_EVERY = 3

# Ink dots for multi-stroke images: dot radius, minimum distance between two dots along a stroke,
# and how many dots the pool holds (spacing * pool size = longest line that can be drawn)
INK_RADIUS = 0.0015
INK_SPACING = 0.002
INK_POOL_SIZE = 1500

# The stamp has to finish inside the shape the marking station drew, so the
# stamped region is the ink shape less this margin.
PAINT_MARGIN = 0.007

# Thickness of the stamp geometry (how tall it sits above the lid).
STAMP_THICKNESS = 0.0008


def _triangle_prism_mesh(shape, thickness, color):
    (x0, y0), (x1, y1), (x2, y2) = shape._corners()[:-1]
    z0, z1 = -thickness / 2, thickness / 2

    vertices = np.array([
        [x0, y0, z0], [x1, y1, z0], [x2, y2, z0],   # bottom face
        [x0, y0, z1], [x1, y1, z1], [x2, y2, z1],   # top face
    ])
    faces = np.array([
        [0, 2, 1], [3, 4, 5],                       # bottom, top
        [0, 1, 4], [0, 4, 3],                       # side 0-1
        [1, 2, 5], [1, 5, 4],                       # side 1-2
        [2, 0, 3], [2, 3, 5],                       # side 2-0
    ])
    solid = trimesh.Trimesh(vertices=vertices, faces=faces)

    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        solid.export(f.name)
        path = f.name

    mesh = Mesh(filename=path, color=color)
    mesh.T = SE3(shape.cx, shape.cy, 0).A
    return mesh

def _triangle_area2(a, b, c):
    # Twice the signed area of triangle abc (positive = counter-clockwise)
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

def _triangulate_polygon(points):
    # Ear clipping: splits a simple polygon (closed or open, either winding) into triangles.
    # Returns (outline, triangles) where outline is counter-clockwise and triangles index into it.
    outline = []
    for x, y in points:
        p = (float(x), float(y))
        if not outline or p != outline[-1]:
            outline.append(p)
    if len(outline) > 1 and outline[0] == outline[-1]:
        outline.pop()                           # drop the repeated closing point
    if len(outline) < 3:
        raise ValueError('_triangulate_polygon: need at least 3 distinct points')

    area = 0.5 * sum(outline[i][0] * outline[(i + 1) % len(outline)][1]
                    - outline[(i + 1) % len(outline)][0] * outline[i][1]
                    for i in range(len(outline)))
    if area == 0:
        raise ValueError('_triangulate_polygon: polygon has no area')
    if area < 0:
        outline.reverse()                       # make it counter-clockwise

    # tolerance relative to the polygon size, so small (metre-scale) shapes are handled correctly
    span = max(max(p[0] for p in outline) - min(p[0] for p in outline),
            max(p[1] for p in outline) - min(p[1] for p in outline))
    eps = 1e-12 * span * span

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # a point on the border of the ear counts as inside, except the ear's own corners
    def blocks(p, a, b, c):
        if p == a or p == b or p == c:
            return False
        return cross(a, b, p) >= -eps and cross(b, c, p) >= -eps and cross(c, a, p) >= -eps

    # drop collinear vertices and zero-width spikes: they add no area but stop the ear search
    remaining = list(range(len(outline)))
    changed = True
    while changed and len(remaining) > 3:
        changed = False
        for k in range(len(remaining)):
            m = len(remaining)
            a = outline[remaining[k - 1]]
            b = outline[remaining[k]]
            c = outline[remaining[(k + 1) % m]]
            if abs(cross(a, b, c)) <= eps:
                remaining.pop(k)
                changed = True
                break

    triangles = []
    while len(remaining) > 3:
        m = len(remaining)
        chosen = None
        for k in range(m):
            i0, i1, i2 = remaining[k - 1], remaining[k], remaining[(k + 1) % m]
            a, b, c = outline[i0], outline[i1], outline[i2]
            if cross(a, b, c) <= eps:
                continue                        # reflex or flat corner, not an ear
            if any(blocks(outline[j], a, b, c) for j in remaining if j not in (i0, i1, i2)):
                continue                        # another vertex sits inside this corner
            chosen = k
            break
        if chosen is None:
            # no clean ear (touching or noisy contour): clip the convex corner with the fewest
            # vertices inside it, so the area is still covered without a fan across a concave shape
            best_k, best_n = None, None
            for k in range(m):
                i0, i1, i2 = remaining[k - 1], remaining[k], remaining[(k + 1) % m]
                a, b, c = outline[i0], outline[i1], outline[i2]
                if cross(a, b, c) <= eps:
                    continue
                n_in = sum(blocks(outline[j], a, b, c) for j in remaining if j not in (i0, i1, i2))
                if best_n is None or n_in < best_n:
                    best_k, best_n = k, n_in
                    if n_in == 0:
                        break
            if best_k is None:
                break                           # no convex corner left, nothing sensible to clip
            chosen = best_k
        i0, i1, i2 = remaining[chosen - 1], remaining[chosen], remaining[(chosen + 1) % m]
        triangles.append((i0, i1, i2))
        remaining.pop(chosen)
    if len(remaining) == 3:
        triangles.append(tuple(remaining))

    # safety check: report when the triangles do not cover the whole polygon
    covered = sum(max(_triangle_area2(outline[a], outline[b], outline[c]), 0.0) / 2
                for a, b, c in triangles)
    if covered < 0.999 * abs(area):
        print(f'_triangulate_polygon: WARNING triangles cover only {100 * covered / abs(area):.1f}% of the shape')
    return outline, triangles


def _polygon_prism_mesh(shape, thickness, color):
    outline, triangles = _triangulate_polygon(shape.points)
    n = len(outline)
    z0, z1 = -thickness / 2, thickness / 2

    vertices = np.array([[x, y, z0] for x, y in outline]
                        + [[x, y, z1] for x, y in outline])
    faces = []
    for a, b, c in triangles:
        # every triangle is forced counter-clockwise, so none can face the wrong way and disappear
        if _triangle_area2(outline[a], outline[b], outline[c]) < 0:
            b, c = c, b
        faces.append([a, c, b])                 # bottom face
        faces.append([a + n, b + n, c + n])     # top face
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])             # side i-j
        faces.append([i, j + n, i + n])
    solid = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)

    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        solid.export(f.name)
        path = f.name

    mesh = Mesh(filename=path, color=color)
    mesh.T = SE3(shape.cx, shape.cy, 0).A
    return mesh

def make_stamp_shape(shape, color=PAINTED, thickness=STAMP_THICKNESS):
    if shape.kind == 'circle':
        return Cylinder(radius=shape.size, length=thickness, color=color)
    if shape.kind == 'square':
        return Cuboid(scale=[2 * shape.size, 2 * shape.size, thickness], color=color)
    if shape.kind == 'triangle':
        return _triangle_prism_mesh(shape, thickness, color)
    if shape.kind == 'complex':
        return _polygon_prism_mesh(shape, thickness, color)
    raise ValueError(f'make_stamp_shape has no geometry for kind={shape.kind!r}')

class MarkedBox:
    def __init__(self, env, x, y, surface_z):
        self.env = env
        self.shape = Cuboid(scale=BOX, pose=SE3(x, y, surface_z + BOX[2] / 2),
                            color=CARDBOARD)
        env.add(self.shape)
        self.marks = []                 # (shape, pose relative to the box)

    @property
    def pose(self):
        return SE3(trnorm(np.asarray(self.shape.T, dtype=float)), check=False)

    def set_pose(self, T):
        self.shape.T = T.A if isinstance(T, SE3) else np.asarray(T)
        here = self.pose
        for shape, offset in self.marks:
            shape.T = (here * offset).A

    def add_mark(self, shape, point_world):
        offset = self.pose.inv() * SE3(float(point_world[0]),
                                    float(point_world[1]),
                                    float(point_world[2]))
        self.marks.append((shape, offset))
        shape.T = (self.pose * offset).A

    # Puts the box back on its stand and forgets all marks. Every mark is parked under the floor:
    # ink marks go back to the pool (MarkPool.reset), stamps stay hidden there.
    # env.remove() is not used because Swift.step() crashes on the removed object afterwards.
    def reset(self, x, y, surface_z, pool=None):
        for shape, _ in self.marks:
            shape.T = MarkPool.PARKED.A
        self.marks = []
        self.shape.T = SE3(x, y, surface_z + BOX[2] / 2).A

class MarkPool:
    PARKED = SE3(0, 0, -1.0)

    def __init__(self, env, count, maker):
        self.items = [maker() for _ in range(count)]
        for item in self.items:
            item.T = self.PARKED.A
            env.add(item)
        self.used = 0

    def take(self):
        if self.used >= len(self.items):
            return None                 # pool exhausted, stop marking
        item = self.items[self.used]
        self.used += 1
        return item

    # Parks every mark that was taken and makes the whole pool available again
    def reset(self):
        for item in self.items[:self.used]:
            item.T = self.PARKED.A
        self.used = 0

class SimulatedPerson:
    SPEED = 0.35
    OUTSIDE_X = CURTAIN_X - 0.65
    INSIDE_X = CURTAIN_X + 0.45

    def __init__(self, env):
        self.x = self.previous_x = self.target_x = self.OUTSIDE_X
        self.y = 0.0
        self.parts = []
        skin = (0.78, 0.57, 0.40, 1)
        shirt = (0.12, 0.45, 0.85, 1)
        trousers = (0.12, 0.16, 0.22, 1)
        for offset, size, colour in [
                ((0, 0, 1.12), (.24, .34, .52), shirt),
                ((0, -.23, 1.07), (.14, .12, .56), skin),
                ((0, .23, 1.07), (.14, .12, .56), skin),
                ((0, -.10, .46), (.16, .15, .80), trousers),
                ((0, .10, .46), (.16, .15, .80), trousers),
                ((.04, -.10, .05), (.26, .16, .10), trousers),
                ((.04, .10, .05), (.26, .16, .10), trousers)]:
            item = Cuboid(scale=size, color=colour)
            self.parts.append((item, np.array(offset), np.array(size) / 2))
        self.parts.append((Sphere(radius=.12, color=skin),
                           np.array([0., 0., 1.52]), np.full(3, .12)))
        for item, _, _ in self.parts:
            env.add(item)
        self._place()

    def _place(self):
        for item, offset, _ in self.parts:
            item.T = SE3(*(offset + [self.x, self.y, 0.])).A

    def request_enter(self, enter):
        self.target_x = self.INSIDE_X if enter else self.OUTSIDE_X

    def step(self, dt):
        self.previous_x = self.x
        distance = self.SPEED * max(0., min(float(dt), .05))
        self.x += float(np.clip(self.target_x - self.x, -distance, distance))
        self._place()

    def blocks_curtain(self):
        # Each beam is tested against swept body-part bounding boxes.
        # Include the previous pose so crossing between updates is not missed.
        for _, offset, half in self.parts:
            lo = offset + [min(self.previous_x, self.x), self.y, 0.] - half
            hi = offset + [max(self.previous_x, self.x), self.y, 0.] + half
            if (lo[0] <= CURTAIN_X + CURTAIN_BEAM_HALF and
                    hi[0] >= CURTAIN_X - CURTAIN_BEAM_HALF and
                    lo[1] <= OPEN_HALF and hi[1] >= -OPEN_HALF and
                    np.any((CURTAIN_BEAM_Z >= lo[2] - CURTAIN_BEAM_HALF) &
                           (CURTAIN_BEAM_Z <= hi[2] + CURTAIN_BEAM_HALF))):
                return True
        return False

    def reset_blocked(self):
        # Simulated occupancy is separate from beam status after a full crossing.
        return self.target_x == self.INSIDE_X or any(
            self.x + offset[0] + half[0] >= CURTAIN_X - CURTAIN_BEAM_HALF
            for _, offset, half in self.parts)


class Swappable:
    PARKED = SE3(0, 0, -1.0)

    def __init__(self, env, poses, maker_a, maker_b):
        self.poses = [SE3(p) if not isinstance(p, SE3) else p for p in poses]
        self.a = [maker_a() for _ in self.poses]
        self.b = [maker_b() for _ in self.poses]
        for group in (self.a, self.b):
            for item in group:
                env.add(item)
        self.show(False)

    def show(self, alarm):
        on, off = (self.b, self.a) if alarm else (self.a, self.b)
        for item, pose in zip(on, self.poses):
            item.T = pose.A
        for item in off:
            item.T = self.PARKED.A

# Visual model of the wrist camera; its pose is updated from arm.cam_pose() while the arm moves
def make_camera_model():
    return Cuboid(scale=[0.05, 0.035, 0.035], color=INK)

def make_arms():
    arm1 = KawasakiPickPlace(base=SE3(-STANDOFF, Y_LOAD, PEDESTAL_H))
    arm2 = StaubliShapeDraw(base=SE3(STANDOFF, Y_MARK, PEDESTAL_H) * SE3.Rz(np.pi))
    arm3 = NachiShapeFill(base=SE3(-STANDOFF, Y_PAINT, PEDESTAL_H))
    return arm1, arm2, arm3

def add_conveyor(env):
    if not os.path.exists(CONVEYOR_DAE):
        print(f'WARNING: {CONVEYOR_DAE} not found - running without the conveyor')
        return
    for k in range(SECTIONS):
        y = (k - (SECTIONS - 1) / 2) * SECTION_LENGTH
        env.add(Mesh(filename=CONVEYOR_DAE, pose=SE3(0, y, 0), color=STEEL))

def add_guarding(env):
    scene = {}
    post = [0.05, 0.05, FENCE_H]
    rail_h = [0.4, 1.0, 1.6]

    corners_x = [CURTAIN_X, FENCE_X]
    for x in corners_x:
        for y in np.linspace(-FENCE_Y, FENCE_Y, 6):
            if x < 0 and abs(y) < OPEN_HALF:
                continue                        # operator opening on the -x side
            env.add(Cuboid(scale=post, pose=SE3(x, y, FENCE_H / 2), color=DARK))
    for y in (-FENCE_Y, FENCE_Y):
        for x in np.linspace(CURTAIN_X, FENCE_X, 5):
            env.add(Cuboid(scale=post, pose=SE3(x, y, FENCE_H / 2), color=DARK))
        for z in rail_h:
            env.add(Cuboid(scale=[FENCE_X - CURTAIN_X, 0.03, 0.03],
                        pose=SE3((CURTAIN_X + FENCE_X) / 2, y, z), color=DARK))
    for z in rail_h:
        env.add(Cuboid(scale=[0.03, FENCE_Y * 2, 0.03], pose=SE3(FENCE_X, 0, z), color=DARK))
        for side in (-1, 1):
            env.add(Cuboid(scale=[0.03, FENCE_Y - OPEN_HALF, 0.03],
                           pose=SE3(CURTAIN_X, side * (FENCE_Y + OPEN_HALF) / 2, z), color=DARK))

    # Light curtain across the operator opening: two columns and the beams
    # between them. Reuse teammate Swappable with the user's opening position.
    for y in (-OPEN_HALF, OPEN_HALF):
        env.add(Cuboid(scale=[0.06, 0.06, FENCE_H], pose=SE3(CURTAIN_X, y, FENCE_H / 2),
                    color=(0.85, 0.72, 0.15, 1)))
    beam_poses = [SE3(CURTAIN_X, 0, z) for z in CURTAIN_BEAM_Z]
    beam = [2 * CURTAIN_BEAM_HALF, 2 * OPEN_HALF, 2 * CURTAIN_BEAM_HALF]
    # Call scene['light_curtain'].show(True) to turn the beams red when broken.
    scene['light_curtain'] = Swappable(
        env, beam_poses,
        lambda: Cuboid(scale=beam, color=YELLOW),
        lambda: Cuboid(scale=beam, color=(0.85, 0.12, 0.12, 1)))

    # User's E-stop stations: operator opening and rear side (visual props).
    for x, y in [(CURTAIN_X - 0.30, -OPEN_HALF - 0.15), (FENCE_X + 0.30, 0.0)]:
        env.add(Cuboid(scale=[0.06, 0.06, 1.0], pose=SE3(x, y, 0.5), color=DARK))
        env.add(Cuboid(scale=[0.12, 0.12, 0.10], pose=SE3(x, y, 1.05), color=YELLOW))
        env.add(Cylinder(radius=0.045, length=0.05, pose=SE3(x, y, 1.125),
                        color=(0.85, 0.12, 0.12, 1)))

    # User's stack light; GUI updates these existing Swappable objects.
    tx, ty = CURTAIN_X - 0.15, OPEN_HALF + 0.15
    env.add(Cylinder(radius=0.02, length=2.0, pose=SE3(tx, ty, 1.0), color=DARK))
    scene['tower'] = {}
    for name, z, lit, dim in [
            ('green', 2.06, (0.15, 0.80, 0.25, 1), (0.08, 0.30, 0.12, 1)),
            ('amber', 2.18, (0.98, 0.62, 0.05, 1), (0.40, 0.28, 0.08, 1)),
            ('red', 2.30, (0.85, 0.12, 0.12, 1), (0.35, 0.10, 0.10, 1))]:
        scene['tower'][name] = Swappable(
            env, [SE3(tx, ty, z)],
            lambda c=dim: Cylinder(radius=0.05, length=0.11, color=c),
            lambda c=lit: Cylinder(radius=0.05, length=0.11, color=c))

    # Keep-out zone painted on the floor rather than drawn as a box in the air,
    # for the same see-through reason.
    scene['keep_out'] = Cuboid(scale=[BELT_WIDTH + 0.55, BELT_LENGTH + 0.25, 0.004],
                            pose=SE3(0, 0, 0.002), color=(0.75, 0.62, 0.10, 1))
    env.add(scene['keep_out'])
    return scene

def build_workcell(env):
    env.add(Cuboid(scale=[5.0, 4.2, GROUND_DEPTH], pose=SE3(-0.9, 0, -GROUND_DEPTH / 2),
                color=(0.30, 0.32, 0.35, 1)))
    scene = add_guarding(env)
    scene['person'] = SimulatedPerson(env)
    add_conveyor(env)

    for x, y in [(-STANDOFF, Y_LOAD), (STANDOFF, Y_MARK), (-STANDOFF, Y_PAINT)]:
        env.add(Cuboid(scale=[PEDESTAL_W, PEDESTAL_W, PEDESTAL_H],
                    pose=SE3(x, y, PEDESTAL_H / 2), color=DARK))

    for (x, y), label in [(STORAGE_XY, 'storage'), (FINISH_XY, 'finish')]:
        env.add(Cuboid(scale=[0.30, 0.30, STAND_TOP],
                    pose=SE3(x, y, STAND_TOP / 2), color=(0.50, 0.46, 0.42, 1)))
        scene[label] = (x, y)

    env.add(Axes(0.30))
    scene['crate'] = Cuboid(scale=CRATE_SIZE, pose=MarkPool.PARKED,
                            color=(0.55, 0.36, 0.20, 1))
    env.add(scene['crate'])
    return scene

