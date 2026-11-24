from xarm7_lib import Robot, RealXArm7
from libcapstone import LightsaberDetector
from robot import Kinematics, XARM7_JOINTS
import time
import numpy as np

kinematics = Kinematics(XARM7_JOINTS)
robot = None

MAX_V = 10  # mm/s
MAX_W = np.pi / 8  # rad/s

# ---- joint limits (radians): verify against your xarm7_lib / the xArm manual ----
NEW_LO = np.array([-2*np.pi, -2.059, -2*np.pi, -0.192, -2*np.pi, -1.693, -2*np.pi]) + 0.02
NEW_HI = np.array([ 2*np.pi,  2.094,  2*np.pi,  3.927,  2*np.pi,  np.pi,  2*np.pi]) - 0.02

# ---- tuning for the new pipeline ----
NEW_LIFT_MM = 150     # extra lift after grasping, so the saber can swing clear of the table
NEW_POS_TOL = 0.5     # mm; raise to 50-100 to let the ee drift if IK can't hold position
NEW_AXIS_TOL = 0.01   # rad on the saber axis


# =====================================================================
# Original functions
# =====================================================================

def matrix_to_axis_angle(rot):
    theta = np.arccos(min(max((rot[0, 0] + rot[1, 1] + rot[2, 2] - 1) / 2, -1), 1))
    if theta == 0 or theta == np.pi:
        return np.array([0, 0, 0])
    v = 1 / (2 * np.sin(theta)) * np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return theta * v


def fk(joints, kinematics):
    last_frame = kinematics.forward_kinematics(joints)[:, :, -1]
    return last_frame[:-1, -1], last_frame[:3, :3]


def jacobian(joints, kinematics):
    frames = kinematics.forward_kinematics(joints)
    pe = frames[:3, 3, -1]
    z = frames[:3, 2, 1:]
    o = frames[:3, 3, 1:]
    dist = pe[:, np.newaxis] - o
    jv = np.cross(z, dist, axis=0)
    jw = z
    return np.vstack((jv, jw))


def ikDiff(robot, kinematics, d):
    maxlinear = np.max(np.abs(d[:3]))
    if not maxlinear: maxlinear = 0.0001
    maxangular = np.max(np.abs(d[3:]))
    if not maxangular: maxangular = 0.0001
    wait = 0.1
    step = min(MAX_V / maxlinear, MAX_W / maxangular)
    for _ in range(int(1 / (step * wait))):
        joints = robot.joint_values
        movement, _, _, _ = np.linalg.lstsq(jacobian(joints, kinematics), d, rcond=None)
        robot.set_velocity(step * movement, duration=0.1)
        time.sleep(wait)
    robot.set_joint_targets(robot.joint_values)


def ik(robot, kinematics, target_pos, target_rot):
    epsilon = 0.1
    curr = np.array(robot.joint_values, dtype=float)  # copy, don't alias robot state
    lr = 0.01
    for _ in range(10000):
        pos, rot = fk(curr, kinematics)
        delta = np.hstack((target_pos - pos, matrix_to_axis_angle(target_rot @ rot.T)))
        if np.linalg.norm(delta) < epsilon:
            break
        movement, _, _, _ = np.linalg.lstsq(jacobian(curr, kinematics), delta, rcond=None)
        curr += movement * lr
    for i in range(7):
        while curr[i] < -np.pi:
            curr[i] += 2 * np.pi
        while curr[i] > np.pi:
            curr[i] -= 2 * np.pi
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
    retry = input("Hit enter to use saved joint angles. Type new to move the arm to new joint angles: ") == "new"
    if retry:
        robot.free_drive()
        input("move the arm slightly above the lightsaber, then hit enter")
        robot.end_free_drive()
        joints = robot.joint_values
        print("Current joint angles:", joints)
        dist = float(input("Enter distance to descend in cm: ")) * 10
    else:
        joints = [-0.3049, 0.321218, 0.294313, 0.632501, -0.319703, 0.321937, 0.258652]
        dist = 70
    robot.set_joint_targets(joints)
    ikDiff(robot, kinematics, np.array([0, 0, -dist, 0, 0, 0]))
    robot.grip(0)
    ikDiff(robot, kinematics, np.array([0, 0, dist, 0, 0, 0]))


def xyz_translation():
    if isinstance(robot, RealXArm7):
        robot.free_drive()
        input("Starting free drive. Hit enter to quit")
        robot.end_free_drive()
        print("ended free drive")
    input("hit enter to move in z")
    ikDiff(robot, kinematics, np.array([0, 0, 100, 0, 0, 0]))
    input("hit enter to move in y")
    ikDiff(robot, kinematics, np.array([0, 100, 0, 0, 0, 0]))
    input("hit enter to move in x")
    ikDiff(robot, kinematics, np.array([100, 0, 0, 0, 0, 0]))
    input("hit enter to rotate by pi/3")
    ikDiff(robot, kinematics, np.array([0, 0, 0, 0, 0, np.pi / 3]))


def tipFrame(color, kinematics):
    # get lightsaber frames
    lsFrames = None
    while lsFrames is None:
        lsFrames = detector.query(color)
        if lsFrames is None:
            time.sleep(0.05)

    # use forward kinematics to get end effector frames
    p_ee, rot_ee = fk(robot.joint_values, kinematics)

    # the saber point further from the end effector is the tip
    diff1 = np.linalg.norm(lsFrames[0] - p_ee)
    diff2 = np.linalg.norm(lsFrames[1] - p_ee)
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

    # T_base->tip
    T_base_tip = np.eye(4)
    T_base_tip[:3, 0] = x
    T_base_tip[:3, 1] = y
    T_base_tip[:3, 2] = z
    T_base_tip[:3, 3] = tip

    # T_ee->tip = inv(T_base->ee) @ T_base->tip
    return np.vstack((np.hstack((rot_ee.T, -rot_ee.T @ p_ee[:, np.newaxis])), np.array([[0, 0, 0, 1]]))) @ T_base_tip


def record_pose(color, n=10, delay=0.1):
    return tipFrame(color, kinematics)


def saber_alignment(T_ee_tip, axis):
    _, rot_ee = fk(robot.joint_values, kinematics)
    saber_dir = rot_ee @ T_ee_tip[:3, 2]
    return abs(np.dot(saber_dir, axis))


def tip_movements(robot):
    """Old version: hardcoded tip targets. Kept for reference; use new_tip_movements."""
    grasp_lightsaber()
    color = input("enter lightsaber color: ").strip()
    Teetp = tipFrame(color, kinematics)

    pos = np.array([572.80119189, -76.42351284, 500.50237615])
    rot = np.eye(3)
    Ttip = np.vstack((np.hstack((rot, pos[:, np.newaxis])), np.array([0, 0, 0, 1])))
    Tee = Ttip @ np.linalg.inv(Teetp)
    vertJoints = ik(robot, kinematics, Tee[:3, 3], Tee[:3, :3])
    robot.set_joint_targets(vertJoints)

    horzPose = np.array([-0.345104, 0.878735, 0.063695, 0.793129, -0.30527, -1.575508, -1.666837])
    pos, _ = fk(horzPose, kinematics)
    print(pos)
    rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    Ttip = np.vstack((np.hstack((rot, pos[:, np.newaxis])), np.array([0, 0, 0, 1])))
    Tee = Ttip @ np.linalg.inv(Teetp)
    horzJoints = ik(robot, kinematics, Tee[:3, 3], Tee[:3, :3])
    robot.set_joint_targets(horzJoints)


def writePlane(detector, color):
    print("start moving the saber")
    c = np.empty((0, 3))
    for i in range(10):
        frames = None
        while frames is None:
            frames = detector.query(color)
        c = np.append(c, [(frames[0] + frames[1]) / 2], axis=0)
        time.sleep(.5)
    centroid = np.mean(c, axis=0)
    _, _, Vt = np.linalg.svd(c - centroid)  # center before fitting
    n = Vt[-1, :]
    d = np.dot(centroid, n)
    plane = np.append(n, d)
    print("n:", n)
    print("centroid:", centroid)
    print("d:", d)
    detector.draw_plane(plane)
    input("press enter to end plane")
    print("end")


# =====================================================================
# Jacobian check (run with mode 'jac'; needs no robot motion)
# =====================================================================

def new_test_jacobian(trials=20):
    """Compare analytic Jacobian to finite differences of fk."""
    worst = 0.0
    for _ in range(trials):
        q = np.random.uniform(-1, 1, 7)
        J = jacobian(q, kinematics)
        eps = 1e-6
        p0, R0 = fk(q, kinematics)
        J_num = np.zeros((6, 7))
        for i in range(7):
            dq = q.copy()
            dq[i] += eps
            p1, R1 = fk(dq, kinematics)
            J_num[:3, i] = (p1 - p0) / eps
            J_num[3:, i] = matrix_to_axis_angle(R1 @ R0.T) / eps
        worst = max(worst, np.abs(J - J_num).max())
    print("max |J - J_numeric| over", trials, "trials:", worst)
    print("OK" if worst < 1e-3 else "MISMATCH: check frame indexing (try 0:7 instead of 1:) or units")


# =====================================================================
# New saber-orientation pipeline
# Grasp -> lift -> measure T_ee_tip once -> rotate the saber axis to
# vertical, then horizontal, holding the ee position.
# =====================================================================

def new_ik(robot, kinematics, target_pos, target_rot, seed=None,
           pos_tol=0.5, rot_tol=0.01, iters=2000, step=0.2):
    """Full-pose limit-aware IK. Returns joint array, or None (with diagnostics) on failure."""
    q = np.clip(np.array(robot.joint_values if seed is None else seed, dtype=float),
                NEW_LO, NEW_HI)
    for _ in range(iters):
        pos, rot = fk(q, kinematics)
        e = np.hstack((target_pos - pos, matrix_to_axis_angle(target_rot @ rot.T)))
        if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
            return q
        J = jacobian(q, kinematics)
        free = np.ones(7, bool)
        for _ in range(7):  # re-solve with violating joints frozen
            dq = np.zeros(7)
            dq[free] = np.linalg.pinv(J[:, free], rcond=1e-3) @ e
            dq *= step
            new = q + dq
            bad = free & ((new < NEW_LO) | (new > NEW_HI))
            if not bad.any():
                break
            free &= ~bad
            dq[~free] = 0
        q = np.clip(q + dq, NEW_LO, NEW_HI)
    pos, rot = fk(q, kinematics)
    e = np.hstack((target_pos - pos, matrix_to_axis_angle(target_rot @ rot.T)))
    print("IK failed. final pos err (mm):", np.linalg.norm(e[:3]),
          " rot err (rad):", np.linalg.norm(e[3:]))
    print("joints at limit:", np.where((q <= NEW_LO + 1e-3) | (q >= NEW_HI - 1e-3))[0] + 1)
    print("seed joints:", np.round(robot.joint_values if seed is None else seed, 3))
    return None


def new_ik_axis(robot, kinematics, target_pos, saber_local, axis, seed,
                pos_tol=NEW_POS_TOL, axis_tol=NEW_AXIS_TOL, iters=3000,
                step=0.5, damping=0.05, verbose=True):
    """Limit-aware IK: hold ee position, align only the saber axis (roll is free)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    seed = np.array(seed, dtype=float)
    q = np.clip(seed, NEW_LO, NEW_HI)
    d0 = fk(q, kinematics)[1] @ saber_local
    if verbose:
        print("initial axis err (deg):", np.degrees(np.arccos(np.clip(d0 @ axis, -1, 1))))
    for _ in range(iters):
        pos, rot = fk(q, kinematics)
        d = rot @ saber_local
        e_pos = (target_pos - pos) / 1000.0                  # meters, so rows match rotation rows
        if d @ axis < -0.999:                                # antiparallel: cross() vanishes, so nudge
            perp = np.cross(d, [1.0, 0.0, 0.0])
            if np.linalg.norm(perp) < 1e-3:
                perp = np.cross(d, [0.0, 1.0, 0.0])
            e_rot = perp / np.linalg.norm(perp)
        else:
            e_rot = np.cross(d, axis)
        if np.linalg.norm(e_pos) * 1000 < pos_tol and np.linalg.norm(e_rot) < axis_tol:
            return q
        e = np.hstack((e_pos, e_rot))
        J = jacobian(q, kinematics).copy()
        J[:3] /= 1000.0                                      # same units as e
        free = np.ones(7, bool)
        for _ in range(7):                                   # freeze joints that would exceed limits
            dq = np.zeros(7)
            Jf = J[:, free]
            dq[free] = Jf.T @ np.linalg.solve(Jf @ Jf.T + damping**2 * np.eye(6), e)
            dq *= step
            bad = free & (((q + dq) < NEW_LO) | ((q + dq) > NEW_HI))
            if not bad.any():
                break
            free &= ~bad
            dq[~free] = 0
        q = np.clip(q + dq, NEW_LO, NEW_HI)
    if verbose:
        pos, rot = fk(q, kinematics)
        d = rot @ saber_local
        print("axis IK failed. final pos err (mm):", np.linalg.norm(target_pos - pos),
              " axis err (deg):", np.degrees(np.arccos(np.clip(d @ axis, -1, 1))))
        print("joints at limit:", np.where((q <= NEW_LO + 1e-3) | (q >= NEW_HI - 1e-3))[0] + 1)
        print("seed joints:", np.round(seed, 3))
    return None


def new_orient_saber(axis, T_ee_tip, seed):
    """Joints that put the saber's long axis along `axis`, holding the ee position."""
    p_ee, _ = fk(seed, kinematics)
    return new_ik_axis(robot, kinematics, p_ee, T_ee_tip[:3, 2], axis, seed)


def new_saber_axis_error_deg(joints, T_ee_tip, axis):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    _, R = fk(joints, kinematics)
    d = R @ T_ee_tip[:3, 2]
    return np.degrees(np.arccos(np.clip(d @ axis, -1, 1)))


def new_measured_axis_error_deg(color, axis):
    """Angle between the detector's saber line and `axis` (sign-agnostic)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    f = detector.query(color)
    if f is None:
        return None
    d = f[0] - f[1]
    d = d / np.linalg.norm(d)
    return np.degrees(np.arccos(np.clip(abs(d @ axis), -1, 1)))


def new_tip_movements():
    grasp_lightsaber()  # ends with the gripper closed and the arm lifted
    color = input("enter lightsaber color: ").strip()

    # extra lift so the saber can swing without hitting the table
    input(f"hit enter to lift a further {NEW_LIFT_MM} mm")
    ikDiff(robot, kinematics, np.array([0, 0, NEW_LIFT_MM, 0, 0, 0]))
    time.sleep(1)  # let the arm settle

    T_ee_tip = tipFrame(color, kinematics)
    if not np.isfinite(T_ee_tip).all():
        print("T_ee_tip contains NaN; check tipFrame / detector output")
        return

    # sanity prints: offset size and how it splits along / across the saber
    off = T_ee_tip[:3, 3]
    along = off @ T_ee_tip[:3, 2]
    perp = np.linalg.norm(off - along * T_ee_tip[:3, 2])
    f = detector.query(color)
    if f is not None:
        print("saber length (detector):", np.linalg.norm(f[0] - f[1]))
    print("ee->tip offset norm:", np.linalg.norm(off), " along saber:", along, " perpendicular:", perp)

    q0 = np.array(robot.joint_values, float)
    for name, axis in [("vertical", [0, 0, 1]), ("horizontal", [0, 1, 0])]:
        q = new_orient_saber(axis, T_ee_tip, seed=q0)
        if q is None:
            print(f"IK failed for {name}")
            return
        print(f"{name}: predicted axis error "
              f"{new_saber_axis_error_deg(q, T_ee_tip, axis):.2f} deg")
        input(f"hit enter to move to {name}")
        robot.set_joint_targets(q)
        time.sleep(3)  # or wait on a motion-complete call, if your library has one
        err = new_measured_axis_error_deg(color, axis)
        if err is not None:
            print(f"{name}: measured axis error {err:.2f} deg (should be near 0)")
        q0 = q  # seed the next solve from this solution


# =====================================================================
# Main
# =====================================================================

detector = LightsaberDetector()
with Robot() as robot:
    robot.register_visualizer(detector)
    input("Hit enter to open the gripper")
    robot.grip(1)
    mode = input("Enter 'xyz' to run the translation sequence, 'grasp' to run the grasp sequence, "
                 "'ik' to run ik positions, 'tip' to run tip movements, 'plane', or 'jac' to test the Jacobian: ").strip()
    if mode == 'plane':
        color = input('enter lightsaber color: ').strip()
        writePlane(detector, color)
    elif mode == "ik":
        run_ik_positions()
    elif mode == "tip":
        new_tip_movements()
    elif mode == "xyz":
        xyz_translation()
    elif mode == "grasp":
        grasp_lightsaber()
    elif mode == "jac":
        new_test_jacobian()
    else:
        print("unknown mode:", mode)