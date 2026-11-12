from xarm7_lib import Robot
#from libcapstone import LightsaberDetector
from robot import Kinematics, XARM7_JOINTS
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
    for i in range(1,len(frames)):
        H = frames[:,:,i-1]
        R, t= H[:3,:3], H[:3,3]
        Z = R@np.array([0,0,1])
        J[3:, i-1] = Z
        J[:3,i-1] = np.cross(Z, (pe-t))
    return J

def pseudo(robot, kinematics):
    J = jacobian(robot, kinematics)
    print(J)
    print(J@J.T)
    JPlus = J.T @np.linalg.inv(J@J.T)
    return JPlus

def ikDiff(robot, kinematics, d, dis):
    step = 10 #in mm
    for i in range(0, dis, step):
        joints = robot.joint_values
        #movement = (pseudo(robot, kinematics) @dir)*step
        movement = step*np.linalg.lstsq(jacobian(robot, kinematics), d)[0]
        #print("MOVE:", movement.shape, movement)
        #print("joints:", joints.shape, joints)
        print("old:",kinematics.forward_kinematics(joints)[:,:,-1])
        print("new:",kinematics.forward_kinematics(joints + movement)[:,:,-1])
        robot.set_joint_targets(joints + movement)



#detector = LightsaberDetector()
with Robot() as robot:
#    robot.register_visualizer(detector)
    kinematics = Kinematics(XARM7_JOINTS)
    print("Starting free drive. Hit 'q' to quit")
    robot.free_drive()
    input("hit the key now")
    robot.end_free_drive()
    print("ended free drive")
    input("hit enter to move in z")
    ikDiff(robot, kinematics, np.array([0,0,1,0,0,0]),100)
    input("hit enter to move in y")
    ikDiff(robot, kinematics, np.array([0,1,0,0,0,0]),100)
    input("hit enter to move in x")
    ikDiff(robot, kinematics, np.array([1,0,0,0,0,0]),100)
    #while True:
     #   print(fk(robot, kinematics))
     #   time.sleep(0.5)
   # print("ended free drive")

    
