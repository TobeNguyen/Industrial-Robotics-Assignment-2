# Including libraries
import os
import time
import swift
import numpy as np
import roboticstoolbox as rtb
from ir_support.robots.DHRobot3D import DHRobot3D
from roboticstoolbox import jtraj
from math import pi

class RS007N(DHRobot3D):
    def __init__(self):
        # DH links
        links = self._create_DH()

        # Names of the robot link files in the directory (colour works for stl files)
        white = (0.90, 0.93, 0.88, 1)   # Munsell 10GY9/1 equivalent — pale yellow-green white
        link3D_names = dict(link0 = 'RS007NLink0', color0 = (0.72, 0.75, 0.70, 1),   # base
                            link1 = 'RS007NLink1', color1 = white,                   # J1 turret
                            link2 = 'RS007NLink2', color2 = white,                   # lower arm
                            link3 = 'RS007NLink3', color3 = white,                   # elbow
                            link4 = 'RS007NLink4', color4 = white,                   # forearm
                            link5 = 'RS007NLink5', color5 = white,                   # wrist
                            link6 = 'RS007NLink6', color6 = (0.35, 0.35, 0.38, 1))   # tool flange
        
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [np.eye(4) for _ in range(7)]

        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_path = os.path.join(current_path, 'RS007N')
        super().__init__(links, link3D_names, name = 'KawasakiRS007N', link3d_dir = mesh_path,
                        qtest = qtest, qtest_transforms = qtest_transforms)
        self.q = qtest

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ('q', 'base') and hasattr(self, '_relation_matrices'):
            self._update_3dmodel()

    def _create_DH(self):
        d = [0.360, 0, 0, 0.375, 0, 0.078]
        a = [0, 0.355, 0, 0, 0, 0]
        alpha = [-pi/2, 0, pi/2, -pi/2, pi/2, 0]
        offset = [0, -pi/2, pi/2, 0, 0, 0]
        qlim_deg = [[-180, 180], [-135, 135], [-155, 155], [-200, 200], [-125, 125], [-360, 360]]
        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i],
                                qlim=np.deg2rad(qlim_deg[i]))
            links.append(link)
        return links
    
def run_swift_demo():
    robot = RS007N()
    env = swift.Swift()
    env.launch(realtime=True)
    robot.add_to_env(env)

    print('Flange at q = 0 (expect 0, 0, 1.168):', np.round(robot.fkine(robot.q).t, 3))

    q_start = robot.q
    q_end = np.array([30, -20, 40, 0, 30, 0]) * np.pi / 180.0
    traj = jtraj(q_start, q_end, 100)
    for q in traj.q:
        robot.q = q
        env.step(0.02)
    time.sleep(2)

if __name__ == "__main__":
    run_swift_demo()