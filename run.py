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
    #start = kinematics.forward_kinematics(robot.joint_values)[:,:,-1]
    #dmat = rot_z(d[5])@rot_y(d[4])@rot_x(d[3])
    #target = dmat @ start
    #target[:3,3] += d[:3]
    #eps_pos = 5
    #eps_rot = 0.1
    step = 0.1
    #print("start:", start)
    #print("target:", target)
    for _ in range(100):
        joints = robot.joint_values
        #end = kinematics.forward_kinematics(joints)[:,:,-1]
        #print(target-end)
        #if np.allclose((target - end)[:3,3], 0.0, rtol=0, atol=eps_pos) and np.allclose((target - end)[:3,:3], 0.0, rtol=0, atol=eps_rot):
        #    break
        movement, _, _, _ = np.linalg.lstsq(jacobian(robot, kinematics), d)
        robot.set_velocity(step * movement)
        time.sleep(0.1)
    robot.set_velocity([0,0,0,0,0,0,0])
    #print("Start:", start)
    #print("End:", end)
    #print("Delta:", end - start)
    #print("Error:", target - end)


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
    ikDiff(robot, kinematics, np.array([0,0,100,0,0,0]))
    input("hit enter to move in y")
    ikDiff(robot, kinematics, np.array([0,100,0,0,0,0]))
    input("hit enter to move in x")
    ikDiff(robot, kinematics, np.array([100,0,0,0,0,0]))
    input("hit enter to rotate by pi/3")
    ikDiff(robot, kinematics, np.array([0,0,0,0,0,np.pi/3]))
    #while True:
     #   print(fk(robot, kinematics))
     #   time.sleep(0.5)
   # print("ended free drive")

    
