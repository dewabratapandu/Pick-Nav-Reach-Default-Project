import time
import math
import numpy as np
import pybullet as p


def tuck_arm_smooth(
    robot_id,
    robot_pos,
    joint_name_to_idx,
    joint_ids,
    duration= 0.5,
    control_hz=240,
    max_force=80.0,
):
    """
    Gently tuck the arm while keeping the gripper closed.

    Parameters
    ----------
    robot_id : int
        PyBullet bodyUniqueId of the robot.
    robot_pos : sequence of float
        Current joint positions, in the same order as `joint_ids`.
    joint_name_to_idx : dict[str, int]
        Mapping from joint name (e.g. 'shoulder_lift_joint') to joint index in PyBullet.
    joint_ids : sequence of int
        Joint indices corresponding to entries in `robot_pos`.
    duration : float
        Time (seconds) over which to tuck the arm.
    control_hz : float
        Control frequency (Hz).
    max_force : float
        Max motor force for the arm and gripper joints during the motion.
    """

    # -------- 1. Define the tuck target for arm joints (you can tweak these) --------
    tuck_targets_name = {
        # conservative tucked-ish pose; adjust to taste
        "shoulder_lift_joint": -1.4,
        "elbow_flex_joint":    2.4,
        "forearm_roll_joint":  0.0,
        "wrist_flex_joint":    -1.0,
        "wrist_roll_joint":    0.0,
    }

    smooth_transition(robot_id, robot_pos, tuck_targets_name, joint_name_to_idx, joint_ids, duration, control_hz, max_force)

    tuck_targets_name = {
        # conservative tucked-ish pose; adjust to taste
        "root_x_axis_joint": -0.5,
    }

    smooth_transition(robot_id, robot_pos, tuck_targets_name, joint_name_to_idx, joint_ids, duration, control_hz, max_force)

    tuck_targets_name = {
        # conservative tucked-ish pose; adjust to taste
        "wrist_flex_joint": 0.8,
    }

    smooth_transition(robot_id, robot_pos, tuck_targets_name, joint_name_to_idx, joint_ids, duration, control_hz, max_force)

def move_robot(robot_id,
    robot_pos,
    target_pos,
    joint_name_to_idx,
    joint_ids,
    duration=0.05,
    control_hz=120,
    max_force=80.0,) :

    target_name_map = {
        # conservative tucked-ish pose; adjust to taste
        "root_x_axis_joint": target_pos[0],
        "root_y_axis_joint": target_pos[1],
        "root_z_rotation_joint": target_pos[2],
    }

    smooth_transition(robot_id, robot_pos, target_name_map, joint_name_to_idx, joint_ids, duration, control_hz,
                      max_force)



def smooth_transition(
    robot_id,
    robot_pos,
    target_joint_map,
    joint_name_to_idx,
    joint_ids,
    duration=2.0,
    control_hz=240,
    max_force=80.0,
):
    """
    Gently tuck the arm while keeping the gripper closed.

    Parameters
    ----------
    robot_id : int
        PyBullet bodyUniqueId of the robot.
    robot_pos : sequence of float
        Current joint positions, in the same order as `joint_ids`.
    joint_name_to_idx : dict[str, int]
        Mapping from joint name (e.g. 'shoulder_lift_joint') to joint index in PyBullet.
    joint_ids : sequence of int
        Joint indices corresponding to entries in `robot_pos`.
    duration : float
        Time (seconds) over which to tuck the arm.
    control_hz : float
        Control frequency (Hz).
    max_force : float
        Max motor force for the arm and gripper joints during the motion.
    """

    # Arm joints we will try to control (only those present in this robot)
    arm_joint_names = [n for n in target_joint_map.keys() if n in joint_name_to_idx]
    if not arm_joint_names:
        print("[tuck_arm] No matching arm joints found in joint_name_to_idx.")
        return

    arm_joint_indices = [joint_name_to_idx[n] for n in arm_joint_names]

    # -------- 2. Also keep gripper closed at its current value --------
    gripper_joint_names = []
    for name in ["r_gripper_finger_joint", "l_gripper_finger_joint"]:
        if name in joint_name_to_idx:
            gripper_joint_names.append(name)

    gripper_joint_indices = [joint_name_to_idx[n] for n in gripper_joint_names]

    # Map joint_id -> index into robot_pos
    id_to_pos_idx = {jid: i for i, jid in enumerate(joint_ids)}

    # Current positions for arm joints (in the same order as arm_joint_names)
    curr_arm_pos = []
    for jname in arm_joint_names:
        jidx = joint_name_to_idx[jname]
        pos_idx = id_to_pos_idx[jidx]
        curr_arm_pos.append(robot_pos[pos_idx])
    curr_arm_pos = np.array(curr_arm_pos, dtype=float)

    # Target positions for arm joints
    target_arm_pos = np.array(
        [target_joint_map[jname] for jname in arm_joint_names],
        dtype=float,
    )

    # Current positions for gripper joints (keep constant)
    curr_gripper_pos = []
    for jname in gripper_joint_names:
        jidx = joint_name_to_idx[jname]
        pos_idx = id_to_pos_idx[jidx]
        curr_gripper_pos.append(robot_pos[pos_idx] - 0.002)
    curr_gripper_pos = np.array(curr_gripper_pos, dtype=float)

    # for each finger link
    for finger_name in ["r_gripper_finger_joint", "l_gripper_finger_joint"]:
        f_idx = joint_name_to_idx[finger_name]
        p.changeDynamics(robot_id, f_idx,
                         lateralFriction=1.5,
                         rollingFriction=0.001,
                         spinningFriction=0.001)

    # -------- 3. Time discretization --------
    steps = max(1, int(duration * control_hz))
    dt = 1.0 / control_hz

    # -------- 4. Interpolate arm joints over time, hold gripper constant --------
    for step in range(steps):
        alpha = (step + 1) / steps  # goes from (1/steps) .. 1
        interp_arm_pos = (1.0 - alpha) * curr_arm_pos + alpha * target_arm_pos

        # Build jointIndices and targetPositions to send in this step
        joint_indices = arm_joint_indices + gripper_joint_indices
        target_positions = interp_arm_pos.tolist() + curr_gripper_pos.tolist()

        max_forces = [max_force] * len(arm_joint_indices)
        for _ in range(len(gripper_joint_indices)):
            max_forces.append(2000000)

        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces= max_forces,
        )

        p.stepSimulation()
        time.sleep(dt)
