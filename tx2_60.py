# Including libraries
import os
import time
import swift
import numpy as np
import roboticstoolbox as rtb
from ir_support.robots.DHRobot3D import DHRobot3D
from roboticstoolbox import jtraj
from math import pi

class TX2_60(DHRobot3D):
    def __init__(self):
        # DH links
        links = self._create_DH()

        # Names of the robot link files in the directory (colour works for stl files)
        staubli_yellow = (0.96, 0.75, 0.05, 1)   # Stäubli signature yellow
        wrist_dark = (0.12, 0.12, 0.12, 1)       # dark wrist joint covers
        flange_silver = (0.65, 0.66, 0.68, 1)    # metallic silver tool flange
        link3D_names = dict(link0 = 'TX2_60Link0', color0 = staubli_yellow,          # base
                            link1 = 'TX2_60Link1', color1 = staubli_yellow,          # shoulder
                            link2 = 'TX2_60Link2', color2 = staubli_yellow,          # upper arm
                            link3 = 'TX2_60Link3', color3 = staubli_yellow,          # elbow
                            link4 = 'TX2_60Link4', color4 = wrist_dark,              # forearm
                            link5 = 'TX2_60Link5', color5 = wrist_dark,              # wrist
                            link6 = 'TX2_60Link6', color6 = flange_silver)           # tool flange
        
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [np.eye(4) for _ in range(7)]

        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_path = os.path.join(current_path, 'TX2_60')

        super().__init__(links, link3D_names, name = 'StaubliTX2_60', link3d_dir = mesh_path,
                        qtest = qtest, qtest_transforms = qtest_transforms)
        self.q = qtest

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ('q', 'base') and hasattr(self, '_relation_matrices'):
            self._update_3dmodel()

    def _create_DH(self):
        d = [0.375, 0, 0.020, 0.310, 0, 0.070]
        a = [0, 0.290, 0, 0, 0, 0]
        alpha = [-pi/2, 0, pi/2, -pi/2, pi/2, 0]
        offset = [0, -pi/2, pi/2, 0, 0, 0]      # makes q = 0 the Staubli zero posture
        qlim_deg = [[-180, 180], [-127.5, 127.5], [-152.5, 152.5],
                    [-270, 270], [-121, 132.5], [-270, 270]]
        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i],
                                qlim=np.deg2rad(qlim_deg[i]))
            links.append(link)
        return links
    
def run_swift_demo():
    robot = TX2_60()
    env = swift.Swift()
    env.launch(realtime=True)
    robot.add_to_env(env)

    print('Flange at q = 0 (expect 0, 0.02, 1.045):', np.round(robot.fkine(robot.q).t, 3))

    q_start = robot.q
    q_end = np.array([30, -20, 40, 0, 30, 0]) * np.pi / 180.0
    traj = jtraj(q_start, q_end, 100)
    for q in traj.q:
        robot.q = q
        env.step(0.02)
    time.sleep(2)

if __name__ == "__main__":
    run_swift_demo()


        



