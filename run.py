from xarm7_lib import Robot
from libcapstone import LightsaberDetector
from robot import Kinematics, XARM7_JOINTS
import keyboard
import time

def fk(robot, kinematics):
    joints = robot.joint_values()
    return kinematics.forward_kinematics(joints)

detector = LightsaberDetector()
with Robot() as robot:
    robot.register_visualizer(detector)
    kinematics = Kinematics(XARM7_JOINTS)
    print("Starting free drive. Hit 'q' to quit")
    robot.free_drive()
    while True:
        print(fk(robot, kinematics))
        if keyboard.is_pressed('q'):
            break
        time.sleep(0.5)
    print("ended free drive")
    robot.end_free_drive()