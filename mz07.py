# Including libraries
import os
import time
import swift
import numpy as np
import roboticstoolbox as rtb
from ir_support.robots.DHRobot3D import DHRobot3D
from roboticstoolbox import jtraj
from math import pi

class MZ07(DHRobot3D):
    def __init__(self):

        # DH links
        links = self._create_DH()

        # Names of the robot link files in the directory (colour works for stl files)
        grey = (0.90, 0.90, 0.88, 1)     # near-white body, matches the datasheet photo
        cover = (0.20, 0.20, 0.20, 1)    # dark wrist covers
        flange = (0.15, 0.15, 0.15, 1)   # near-black tool flange
        link3D_names = dict(link0 = 'MZ07Link0', color0 = grey,     # base
                            link1 = 'MZ07Link1', color1 = grey,     # J1 turret
                            link2 = 'MZ07Link2', color2 = grey,     # lower arm
                            link3 = 'MZ07Link3', color3 = cover,    # elbow casting
                            link4 = 'MZ07Link4', color4 = cover,    # forearm
                            link5 = 'MZ07Link5', color5 = cover,    # wrist
                            link6 = 'MZ07Link6', color6 = flange)   # tool flange

        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [np.eye(4) for _ in range(7)]

        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_path = os.path.join(current_path, 'MZ07')
        super().__init__(links, link3D_names, name = 'NachiMZ07', link3d_dir = mesh_path,
                        qtest = qtest, qtest_transforms = qtest_transforms)
        self.q = qtest

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ('q', 'base') and hasattr(self, '_relation_matrices'):
            self._update_3dmodel()

    def _create_DH(self):
        d = [0.345, 0, 0, 0.340, 0, 0.083]
        a = [0.050, 0.330, 0.045, 0, 0, 0]
        alpha = [pi/2, 0, pi/2, pi/2, -pi/2, 0]
        offset = [0, pi/2, 0, 0, 0, 0]
        qlim_deg = [[-170, 170], [-135, 80], [-136, 270], [-190, 190], [-120, 120], [-360, 360]]
        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i],
                                qlim=np.deg2rad(qlim_deg[i]))
            links.append(link)
        return links


def run_swift_demo():
    robot = MZ07()
    env = swift.Swift()
    env.launch(realtime=True)
    robot.add_to_env(env)

    print('Flange at q = 0 (expect 0.473, 0, 0.720):', np.round(robot.fkine(robot.q).t, 3))

    q_start = robot.q
    q_end = np.array([30, -20, 40, 0, 30, 0]) * np.pi / 180.0
    traj = jtraj(q_start, q_end, 100)
    for q in traj.q:
        robot.q = q
        env.step(0.02)
    time.sleep(2)


if __name__ == "__main__":
    run_swift_demo()
