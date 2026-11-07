from xarm7_lib import Robot
#from libcapstone import LightsaberDetector
from robot import Kinematics, XARM7_JOINTS
import time
import numpy as np

def fk(robot, kinematics):
    joints = robot.joint_values
    return kinematics.forward_kinematics(joints)[:,:,-1] @ np.array([[0],[0],[0],[1]])

#detector = LightsaberDetector()
with Robot() as robot:
#    robot.register_visualizer(detector)
    kinematics = Kinematics(XARM7_JOINTS)
    print("Starting free drive. Hit 'q' to quit")
    robot.free_drive()
    while True:
        print(fk(robot, kinematics))
        time.sleep(0.5)
    print("ended free drive")
    robot.end_free_drive()
