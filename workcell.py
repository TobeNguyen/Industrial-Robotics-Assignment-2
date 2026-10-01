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

CONVEYOR_DAE = os.path.join(current_path, 'environment', 'conveyor_belt.dae')
SECTION_LENGTH = 1.200
SECTIONS = 2                            # butted end to end -> 2.400 m of belt
BELT_Z = 0.741
BELT_WIDTH = 0.463
BELT_LENGTH = SECTIONS * SECTION_LENGTH

# Robot mounting=
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
SHAPE_SIZE = 0.035

# Safety Equipment
FENCE_X, FENCE_Y, FENCE_H = 1.400, 1.750, 1.800

STEEL = (0.62, 0.65, 0.68, 1)
DARK = (0.42, 0.44, 0.46, 1)
YELLOW = (0.85, 0.72, 0.15, 1)
CARDBOARD = (0.76, 0.60, 0.38, 1)
PAINTED = (0.20, 0.45, 0.78, 1)         # spray/stamp colour
INK = (0.10, 0.10, 0.12, 1)

INK_EVERY = 3

# The stamp has to finish inside the shape the marking station drew
PAINT_MARGIN = 0.007

# Thickness of the stamp geometry (how tall it sits above the lid).
STAMP_THICKNESS = 0.0008


def _triangle_prism_mesh(shape, thickness):
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

    mesh = Mesh(filename=path, color=PAINTED)
    mesh.T = SE3(shape.cx, shape.cy, 0).A
    return mesh


def make_stamp_shape(shape, thickness=STAMP_THICKNESS):
    if shape.kind == 'circle':
        return Cylinder(radius=shape.size, length=thickness, color=PAINTED)
    if shape.kind == 'square':
        return Cuboid(scale=[2 * shape.size, 2 * shape.size, thickness], color=PAINTED)
    if shape.kind == 'triangle':
        return _triangle_prism_mesh(shape, thickness)
    raise ValueError(f'make_stamp_shape has no geometry for kind={shape.kind!r}')

class MarkedBox:
    def __init__(self, env, x, y, surface_z):
        self.env = env
        self.shape = Cuboid(scale=BOX, pose=SE3(x, y, surface_z + BOX[2] / 2),
                            color=CARDBOARD)
        env.add(self.shape)
        self.marks = []

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

class MarkPool:
    def __init__(self, env, count, maker):
        self.items = [maker() for _ in range(count)]
        for item in self.items:
            item.T = SE3(0, 0, -1.0).A
            env.add(item)
        self.used = 0

    def take(self):
        if self.used >= len(self.items):
            return None                 
        item = self.items[self.used]
        self.used += 1
        return item

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

    corners_x = [-FENCE_X, FENCE_X]
    for x in corners_x:
        for y in np.linspace(-FENCE_Y, FENCE_Y, 6):
            if x < 0 and abs(y) < FENCE_Y * 0.8:
                continue                        # operator opening on the -x side
            env.add(Cuboid(scale=post, pose=SE3(x, y, FENCE_H / 2), color=DARK))
    for y in (-FENCE_Y, FENCE_Y):
        for x in np.linspace(-FENCE_X, FENCE_X, 5):
            env.add(Cuboid(scale=post, pose=SE3(x, y, FENCE_H / 2), color=DARK))
        for z in rail_h:
            env.add(Cuboid(scale=[FENCE_X * 2, 0.03, 0.03], pose=SE3(0, y, z), color=DARK))
    for z in rail_h:
        env.add(Cuboid(scale=[0.03, FENCE_Y * 2, 0.03], pose=SE3(FENCE_X, 0, z), color=DARK))

    # Light curtain across the operator opening: two columns and the beams
    # between them. scene['light_curtain'] is the list the safety code will
    # recolour when the curtain is broken.
    for y in (-FENCE_Y * 0.8, FENCE_Y * 0.8):
        env.add(Cuboid(scale=[0.06, 0.06, FENCE_H], pose=SE3(-FENCE_X, y, FENCE_H / 2),
                    color=(0.85, 0.72, 0.15, 1)))
    beam_poses = [SE3(-FENCE_X, 0, z) for z in np.linspace(0.25, FENCE_H - 0.25, 7)]
    beam = [0.012, FENCE_Y * 1.6, 0.012]
    
    scene['light_curtain'] = Swappable(
        env, beam_poses,
        lambda: Cuboid(scale=beam, color=YELLOW),
        lambda: Cuboid(scale=beam, color=(0.85, 0.12, 0.12, 1)))

    # Keep-out zone painted on the floor rather than drawn as a box in the air,
    # for the same see-through reason.
    scene['keep_out'] = Cuboid(scale=[BELT_WIDTH + 0.55, BELT_LENGTH + 0.25, 0.004],
                            pose=SE3(0, 0, 0.002), color=(0.75, 0.62, 0.10, 1))
    env.add(scene['keep_out'])
    return scene

def build_workcell(env):
    env.add(Cuboid(scale=[3.4, 4.2, 0.01], pose=SE3(0, 0, -0.005), color=(0.30, 0.32, 0.35, 1)))
    scene = add_guarding(env)
    add_conveyor(env)

    for x, y in [(-STANDOFF, Y_LOAD), (STANDOFF, Y_MARK), (-STANDOFF, Y_PAINT)]:
        env.add(Cuboid(scale=[PEDESTAL_W, PEDESTAL_W, PEDESTAL_H],
                    pose=SE3(x, y, PEDESTAL_H / 2), color=DARK))

    for (x, y), label in [(STORAGE_XY, 'storage'), (FINISH_XY, 'finish')]:
        env.add(Cuboid(scale=[0.30, 0.30, STAND_TOP],
                    pose=SE3(x, y, STAND_TOP / 2), color=(0.50, 0.46, 0.42, 1)))
        scene[label] = (x, y)

    env.add(Axes(0.30))
    return scene

