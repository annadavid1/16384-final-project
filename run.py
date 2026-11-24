from xarm7_lib import Robot, RealXArm7
from libcapstone import LightsaberDetector
from robot import Kinematics, XARM7_JOINTS, rot_x, rot_y, rot_z, translation
import time
import numpy as np

kinematics = Kinematics(XARM7_JOINTS)
robot = None

MAX_V = 10 # mm/s
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
        robot.set_velocity(step * movement, duration = 0.1)
        time.sleep(wait)
    #robot.set_velocity([0,0,0,0,0,0,0])
    robot.set_joint_targets(robot.joint_values)

def ik(robot, kinematics, target_pos, target_rot):
    epsilon_pos = 0.1
    epsilon_rot = 0.01
    curr = robot.joint_values
    lr = 0.01
    limits = robot.joint_limits
    for _ in range(10000):
        pos, rot = fk(curr, kinematics)
        delta = np.hstack((target_pos - pos, matrix_to_axis_angle(target_rot @ rot.T)))
        if np.linalg.norm(delta[:3]) < epsilon_pos and np.linalg.norm(delta[3:]) < epsilon_rot:
            break
        movement, _, _, _ = np.linalg.lstsq(jacobian(curr, kinematics), delta)
        update = movement * lr
        curr += update
        for i in range(7):
            curr[i] = np.clip(
                curr[i],
                limits[i, 0],
                limits[i, 1]
            )
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
    robot.set_joint_targets(joints)
    input("Hit enter to run IK for position 2")
    joints = ik(robot, kinematics, pos2, rot2)
    robot.set_joint_targets(joints)
    input("Hit enter to run IK for position 3")
    joints = ik(robot, kinematics, pos3, rot3)
    robot.set_joint_targets(joints)

def grasp_lightsaber():
    input("Hit enter to open the gripper")
    robot.grip(1)
    retry = input("Hit enter to use saved joint angles. Type new to move the arm to new joint angles: ") == "new"
    if retry:
        robot.free_drive()
        input("move the arm slightly above the lightsaber, then hit enter")
        joints = robot.joint_values
        print("Current joint angles:", joints)
        dist = float(input("Enter distance to descend in cm: ")) * 10
    else:
        joints = [0, 0, 0, 0, 0, 0, 0] # [-0.3049, 0.321218, 0.294313, 0.632501, -0.319703, 0.321937, 0.258652]
        dist = 10
    robot.set_joint_targets(joints)
    ikDiff(robot, kinematics, np.array([0,0,-dist,0,0,0]))
    robot.grip(0)
    ikDiff(robot, kinematics, np.array([0,0,dist+50,0,0,0]))

def xyz_translation():
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

def tipFrame(color, kinematics):
    #get lightsaber frames
    lsFrames = None
    while lsFrames is None:
        print("HELP")
        lsFrames = detector.query(color)
        print("lsFrames", lsFrames)
    print(lsFrames.shape)
    lsFrames *= 1000
    #use forward kinematics to get end effector frames
    p_ee, rot_ee = fk(robot.joint_values,kinematics)

    #find which lightsaber frame is further from the end effector frame
    diff1 = np.linalg.norm(lsFrames[0] - p_ee)
    diff2 = np.linalg.norm(lsFrames[1] - p_ee)

    #find transform ee->tip by doing T^-1 (base ->ee) * T (base->tip)
    if diff1 > diff2:
        tip, handle = lsFrames[0], lsFrames[1]
    else:
        tip, handle = lsFrames[1], lsFrames[0]

    # z axis: handle -> tip
    z = tip - handle
    z = z / np.linalg.norm(z)

    # y axis: ee's y axis projected perpendicular to z
    y_ee = rot_ee[:, 1]
    y = y_ee - np.dot(y_ee, z) * z
    y = y / np.linalg.norm(y)

    # x axis completes right-handed frame
    x = np.cross(y, z)

    # build T_base->tip
    T_base_tip = np.eye(4)
    T_base_tip[:3, 0] = x
    T_base_tip[:3, 1] = y
    T_base_tip[:3, 2] = z
    T_base_tip[:3, 3] = tip

    print(T_base_tip[:3,:3] @ T_base_tip[:3,:3].T)

    # T_ee->tip = inv(T_base->ee) @ T_base->tip
    return np.vstack((np.hstack((rot_ee.T, -rot_ee.T @ p_ee[:, np.newaxis])), np.array([[0,0,0,1]]))) @ T_base_tip

def tip_movements(robot):
    color = None
    # grasp_lightsaber()
    #lightsaber motion portion
    color = input("enter lightsaber color: ").strip()

    Teetp = tipFrame(color, kinematics)

    #get into initial postition and find joint angles
    input("hit enter to move 200mm in z")
    ikDiff(robot, kinematics, np.array([0,0,200,0,0,0]))
    time.sleep(1)
    eeJoints = np.array(robot.joint_value,float)

    #find joint angles necessary for vertical
    eePos = fk(eeJoints, kinematics)
    saberForm =  Teetp[:3,2] * -1 #CHANGE FOR MILESTONE 3, hardcoding which way saber is pointed

    #this bit might be weird
    d0 = fk(eePos, kinematics)[1] @ saberForm
    vertJoints = ik(robot, kinematics, d0, np.array[0,0,1])
    robot.set_joint_targets(vertJoints)
    time.sleep(10)

    #find stuff for horizontal
    horzJoints = ik(robot, kinematics, d0, np.array[0,1,0])
    robot.set_joint_targets(vertJoints)
    time.sleep(10)


    # Ttip = np.vstack((np.hstack((rot, pos[:,np.newaxis])), np.array([0,0,0,1])))
    # Ttpee = np.linalg.inv(Teetp)
    # Tee = Ttip @ Ttpee
    # print(Tee)
    # vertJoints = ik(robot,kinematics,Tee[:3, 3], Tee[:3,:3])
    # robot.set_joint_targets(vertJoints)
    # print(pos)

    # horzPose = np.array([-0.345104,  0.878735,  0.063695,  0.793129, -0.30527,  -1.575508, -1.666837])
    # pos, _ = fk(horzPose, kinematics)
    # # pos = np.array([ 280.38120849, -306.56541326,   17.77073711])
    # print(pos)
    # rot = np.array([[1, 0, 0],[0, 0, 1],[0, -1, 0]]) # -90 degrees about x
    # Ttip = np.vstack((np.hstack((rot, pos[:,np.newaxis])), np.array([0,0,0,1])))
    # Ttpee = np.linalg.inv(Teetp)
    # Tee = Ttip @ Ttpee
    # horzJoints = ik(robot,kinematics,Tee[:3, 3], Tee[:3,:3])
    # robot.set_joint_targets(horzJoints)

def writePlane(detector, color):
    c = np.empty((0,3))
    for i in range(0,10,1):
        frames = None
        while frames is None:
            frames = detector.query(color)
        c= np.append(c,[(frames[0] + frames[1])/2], axis =0)
        time.sleep(.5)
    #center =abs(c-np.mean(c, axis=0))
    center = c
    _,_,V= np.linalg.svd(center)
    n=V[-1,:]
    centroid = np.mean(center, axis=0)
    print("n:", n)
    print(center)
    print("centroid:", centroid)
    d= np.dot(centroid, n)
    plane = np.append(n,d)
    print(d)
    detector.draw_plane(plane)
    end = input("press enter to end plane")
    if end != None: print("end")


        

detector = LightsaberDetector()
with Robot() as robot:
    robot.register_visualizer(detector)
    mode = input("Enter 'xyz' to run the translation sequence or 'grasp' to run the grasp sequence, or 'ik' to run ik positions, or 'tip' to run tip movements or 'plane': ").strip()
    if mode == 'plane':
        color = input('enter lightsaber color: ').strip()
        writePlane(detector, color)
    elif mode == "ik":
        run_ik_positions()
    elif mode == "tip":
        tip_movements(robot)
    elif mode == "xyz":
        xyz_translation()
    elif mode == "grasp":
        grasp_lightsaber()

    #writePlane(detector, color)


    #while True:
     #   print(fk(robot, kinematics))
     #   time.sleep(0.5)
   # print("ended free drive")

    
