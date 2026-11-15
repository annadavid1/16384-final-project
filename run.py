from xarm7_lib import Robot
#from libcapstone import LightsaberDetector
from robot import Kinematics, XARM7_JOINTS, rot_x, rot_y, rot_z, translation
import time
import numpy as np

def fk(robot, kinematics):
    joints = robot.joint_values
    return kinematics.forward_kinematics(joints)[:,:,-1] @ np.array([[0],[0],[0],[1]])

def jacobian(robot, kinematics):
    joints = robot.joint_values
    frames = kinematics.forward_kinematics(joints)
    J = np.zeros([6,7])
    pe = frames[:3,3, -1]
    for i in range(frames.shape[2]-1):
        H = frames[:,:,i+1]
        R = H[:3,:3]
        t = H[:3,3]
        Z = R@np.array([0,0,1])
        J[3:, i] = Z
        J[:3,i] = np.cross(Z, (pe-t))
    return J

def pseudo(robot, kinematics):
    J = jacobian(robot, kinematics)
    print(J)
    print(J@J.T)
    JPlus = J.T @np.linalg.inv(J@J.T)
    return JPlus

def ikDiff(robot, kinematics, d):
    wait = 0.1
    step = 0.1
    for _ in range(int(1/(step*wait))):
        movement, _, _, _ = np.linalg.lstsq(jacobian(robot, kinematics), d)
        robot.set_velocity(step * movement)
        time.sleep(0.1)
    robot.set_velocity([0,0,0,0,0,0,0])



#detector = LightsaberDetector()
with Robot() as robot:
#    robot.register_visualizer(detector)
    kinematics = Kinematics(XARM7_JOINTS)
    mode = input("Enter 'xyz' to run the translation sequence or 'grasp' to run the grasp sequence: ")
    if mode == "xyz":
        robot.free_drive()
        input("Starting free drive. Hit enter to quit")
        robot.end_free_drive()
        print("ended free drive")
        input("hit enter to move in z")
        ikDiff(robot, kinematics, np.array([0,0,100,0,0,0]))
        input("hit enter to move in y")
        ikDiff(robot, kinematics, np.array([0,100,0,0,0,0]))
        input("hit enter to move in x")
        ikDiff(robot, kinematics, np.array([100,0,0,0,0,0]))
        input("hit enter to rotate by pi/3")
        ikDiff(robot, kinematics, np.array([0,0,0,0,0,np.pi/3]))
    elif mode == "grasp":
        robot.free_drive()
        input("move the arm slightly above the lightsaber, then hit enter")
        joints = robot.joint_angles
        print("Current joint angles:", joints)
        dist = input("Enter distance to descend in cm: ") * 10
        robot.set_joint_targets(joints)
        ikDiff(robot, kinematics, np.array([0,0,-dist,0,0,0]))
        robot.grip(0)
        ikDiff(robot, kinematics, np.array([0,0,dist,0,0,0]))
    #while True:
     #   print(fk(robot, kinematics))
     #   time.sleep(0.5)
   # print("ended free drive")

    
