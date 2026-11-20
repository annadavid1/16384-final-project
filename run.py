from xarm7_lib import Robot, RealXArm7
#from libcapstone import LightsaberDetector
from robot import Kinematics, XARM7_JOINTS, rot_x, rot_y, rot_z, translation
import time
import numpy as np


MAX_V = 20 # mm/s
MAX_W = np.pi/8 # rad/s

def matrix_to_axis_angle(rot):
    theta = np.acos(min(max((rot[0,0]+rot[1,1]+rot[2,2]-1)/2, -1), 1))
    if theta == 0 or theta == np.pi:
        return np.array([0, 0, 0])
    v = 1/(2*np.sin(theta))*np.array([rot[2,1]-rot[1,2], rot[0,2]-rot[2,0], rot[1,0]-rot[0,1]])
    return theta * v

def fk(joints, kinematics):
    last_frame = kinematics.forward_kinematics(joints)[:,:,-1]
    return last_frame[:-1,-1], last_frame[:3,:3]

def jacobian(joints, kinematics):
    frames = kinematics.forward_kinematics(joints)
    J = np.zeros([6,7])
    pe = frames[:3,3, -1]
    z = frames[:3,2,1:]
    o = frames[:3,3,1:]
    dist = pe[:, np.newaxis] - o
    jv = np.cross(z, dist, axis=0)
    jw = z
    J = np.vstack((jv, jw))
    return J

def ikDiff(robot, kinematics, d):
    maxlinear = np.max(np.abs(d[:3]))
    if not maxlinear: maxlinear = 0.0001
    maxangular = np.max(np.abs(d[3:]))
    if not maxangular: maxangular = 0.0001
    wait = 0.1
    step = min(MAX_V / maxlinear, MAX_W / maxangular)
    for _ in range(int(1/(step*wait))):
        joints = robot.joint_values
        movement, _, _, _ = np.linalg.lstsq(jacobian(joints, kinematics), d)
        robot.set_velocity(step * movement)
        time.sleep(wait)
    robot.set_velocity([0,0,0,0,0,0,0])

def ik(robot, kinematics, target_pos, target_rot):
    epsilon = 0.1
    curr = robot.joint_values
    lr = 0.01
    for i in range(10000):
        pos, rot = fk(curr, kinematics)
        delta = np.hstack((target_pos - pos, matrix_to_axis_angle(target_rot @ rot.T)))
        print(matrix_to_axis_angle(target_rot @ rot.T))
        print(delta)
        if np.linalg.norm(delta) < epsilon:
            break
        movement, _, _, _ = np.linalg.lstsq(jacobian(curr, kinematics), delta)
        update = movement * lr
        curr += update
    return curr

def run_ik_positions():
    pos1 = np.array([500, 0, 0])
    rot1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    pos2 = np.array([600, 0, 300])
    rot2 = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    pos3 = np.array([400, 200, 250])
    rot3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    input("Hit enter to run IK for position 1")
    joints = ik(robot, kinematics, pos1, rot1)
    robot.set_joint_angles(joints)
    input("Hit enter to run IK for position 2")
    joints = ik(robot, kinematics, pos2, rot2)
    robot.set_joint_angles(joints)
    input("Hit enter to run IK for position 3")
    joints = ik(robot, kinematics, pos3, rot3)
    robot.set_joint_angles(joints)


#detector = LightsaberDetector()
with Robot() as robot:
#    robot.register_visualizer(detector)
    kinematics = Kinematics(XARM7_JOINTS)
    input("Hit enter to open the gripper")
    robot.grip(1)
    mode = input("Enter 'xyz' to run the translation sequence or 'grasp' to run the grasp sequence, or 'ik' to run ik positions: ").strip()
    if mode == "ik":
        run_ik_positions()
    elif mode == "xyz":
        if isinstance(robot, RealXArm7):
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
        retry = input("Hit enter to use saved joint angles. Type new to move the arm to new joint angles: ") == "new"
        if retry:
            robot.free_drive()
            input("move the arm slightly above the lightsaber, then hit enter")
            joints = robot.joint_values
            print("Current joint angles:", joints)
            dist = float(input("Enter distance to descend in cm: ") * 10)
        else:
            joints = [-0.3049, 0.321218, 0.294313, 0.632501, -0.319703, 0.321937, 0.258652]
            dist = 50
        robot.set_joint_targets(joints)
        ikDiff(robot, kinematics, np.array([0,0,-dist,0,0,0]))
        robot.grip(0)
        ikDiff(robot, kinematics, np.array([0,0,dist,0,0,0]))
        input("Hit enter to release the lightsaber")
        robot.grip(1)
    #while True:
     #   print(fk(robot, kinematics))
     #   time.sleep(0.5)
   # print("ended free drive")

    
