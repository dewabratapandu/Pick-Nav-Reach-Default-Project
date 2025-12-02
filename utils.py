import numpy as np

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def closest_joint_values(target_joint_values, current_joint_values, wrap_mask):
    """
    Args:
        target_joint_values: np.ndarray, shape (N,)
        current_joint_values: np.ndarray, shape (N,)
        wrap_mask: np.ndarray[bool or int], shape (N,)  # 1 表示需要 wrap, 0 表示直接保留

    Returns:
        output_joint_values: np.ndarray, shape (N,)
    """
    target_joint_values = np.asarray(target_joint_values, dtype=float)
    current_joint_values = np.asarray(current_joint_values, dtype=float)
    wrap_mask = np.asarray(wrap_mask, dtype=bool)

    output_joint_values = target_joint_values.copy()

    diffs = wrap_to_pi(target_joint_values[wrap_mask] - current_joint_values[wrap_mask])
    output_joint_values[wrap_mask] = current_joint_values[wrap_mask] + diffs

    return output_joint_values


import pybullet as p
import time

def tuck_arm(robot_pos, joint_name_to_idx, joint_ids):
    """
    Move the arm into a compact pose.

    Parameters
    ----------
    robot_id : int
        Body unique ID from p.loadURDF.
    joint_name_to_idx : dict[str, int]
        Mapping from joint name -> joint index (build once using getJointInfo).
    wait : bool
        If True, will step the simulation for a bit so the arm settles.
        If you're already stepping in your main loop, set to False and just call once.
    steps : int
        Number of sim steps to wait if wait=True.
    """

    #Altering shoulder and elbow values
    target_by_name = {
        "shoulder_lift_joint":    -1.4,    # arm a bit up
        "elbow_flex_joint":       2.4,    # elbow bent
        "wrist_flex_joint":    1.5,
    }
    print(f"Joint dictionary: {joint_name_to_idx}")
    updated_robot_pos = robot_pos.copy()
    for jname, jpos in target_by_name.items():
        if jname in joint_name_to_idx:  # in case some joints are missing/renamed
            j_idx = joint_name_to_idx[jname]
            #look for id position in joint_ids
            idx = joint_ids.index(j_idx)
            updated_robot_pos[idx] = jpos

    return updated_robot_pos

# Reset the debug visualizer camera
def set_camera_on_robot(curr_robot_x, curr_robot_y):
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,  # Distance of the camera from the target
        cameraYaw=30.0,  # Yaw angle (horizontal rotation) in degrees
        cameraPitch=-30.0,  # Pitch angle (vertical rotation) in degrees
        cameraTargetPosition=[curr_robot_x, curr_robot_y, 1],
    )
