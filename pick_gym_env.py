from time import time
import pybullet as p
import pybullet_data
import numpy as np
import random 
import trimesh 
from pathlib import Path

import utils
from maze_utils import generate_maze_map, add_left_room_to_maze, create_maze_urdf
from copy import deepcopy
from keyboard_control import KeyBoardController
from robot_auxiliary_movements import tuck_arm_smooth
from utils import closest_joint_values, tuck_arm
import path_planner as path_planner
import time
import gymnasium as gym
from stable_baselines3 import SAC

class PickEnv(gym.Env):

    def __init__(self,
                 seed=0,
                 gui=False,
                 object_idx=5,
                 use_astar = False
                 ):
        self.set_seed(seed)

        #for training use p.DIRECT
        gui_var = p.GUI if gui else p.DIRECT
        self.pb_physics_client = p.connect(gui_var)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.pb_physics_client)
        p.setGravity(0,0,-9.8, physicsClientId=self.pb_physics_client)

        p.configureDebugVisualizer(lightPosition=[0, 0, 10], physicsClientId=self.pb_physics_client)
        p.configureDebugVisualizer(rgbBackground=[1, 1, 1], physicsClientId=self.pb_physics_client)        # white background brightens perception

        self.seed = seed
        #TODO:randomize later
        self.object_idx = object_idx

        self.max_steps = 100

        
        self.action_scale = 0.05
        self.max_force = 2000000
        self.substeps = 5
        self.debug_point_id = None
        self.debug_line_ids = []
        self.use_astar = use_astar

        # loading env
        p.loadURDF("plane.urdf", physicsClientId=self.pb_physics_client)
        self._load_scene()
        #for _ in range(100):  # tune: 50–200 steps
        #    p.stepSimulation(physicsClientId=self.pb_physics_client)

        #self._reset_robot()

        # define action space, 9 joint values
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.ctrl_joints),),
            dtype=np.float32,
        )

        #observation space - 3 end-effector position, ee-yaw, 3 object position, 3-relative positioning, grip_width, 9-joint values
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32
        )
        
        # get initial observation
        obs = self._get_obs_gym()
        
        self.step_count = 0
        

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
    
    def _load_scene(self):
        self._load_agent()
        self._load_object_table()
        self._load_maze()
        self._load_object_goal()

        #Let it settle a bit
        #for _ in range(20):
            #p.stepSimulation(physicsClientId=self.pb_physics_client)

    def _load_agent(self):
        # Place the Fetch base near the table
        base_pos = [-1., 0.0, 0.0]
        base_ori = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.pb_physics_client)
        self.robot_id = p.loadURDF(
            "assets/fetch/fetch.urdf",
            base_pos,
            base_ori,
            useFixedBase=True,
            physicsClientId=self.pb_physics_client,
            # flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        )

        # Collect joint info (skip fixed)
        n_joints = p.getNumJoints(self.robot_id, physicsClientId=self.pb_physics_client)
        indices = []
        lowers, uppers, ranges, rest, name_to_id = [], [], [], [], {}

        for j in range(n_joints):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.pb_physics_client)
            joint_type = info[2]
            name_to_id[info[1].decode("utf-8")] = j
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                indices.append(j)
                lowers.append(info[8])
                uppers.append(info[9])
                ranges.append(info[9] - info[8])
                rest.append(info[10])  # joint damping? (PyBullet packs different things; we keep a placeholder)

                p.setJointMotorControl2(
                    self.robot_id, j, p.VELOCITY_CONTROL, force=0.0, physicsClientId=self.pb_physics_client
                )

        self.joint_indices = indices
        self.joint_lower = np.array(lowers, dtype=np.float32)
        self.joint_upper = np.array(uppers, dtype=np.float32)
        self.joint_ranges = np.array(ranges, dtype=np.float32)
        self.rest_poses = np.zeros_like(self.joint_lower, dtype=np.float32)
        self.joint_name_to_id = name_to_id

        #setting joint info
        # Arm DOFs:
        self.arm_joints = [13, 14, 15, 16, 17, 18, 19]

        # Gripper:
        self.gripper_joint_indices = [21, 22]

        # Joints controlled by RL (arm + gripper)
        self.ctrl_joints = self.arm_joints + self.gripper_joint_indices  # 9 joints

        # Joints not controlled by RL (arm + gripper)
        self.non_ctrl_joints = [j for j in self.joint_indices if j not in self.ctrl_joints]

        #end-effector
        self.ee_link_index = 20

        # Set an initial configuration
        init_qpos = np.clip(
            np.array([0.0] * len(self.joint_indices)),
            self.joint_lower,
            self.joint_upper,
        )

        crane_pose = {
            "torso_lift_joint": 0.15,
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.,
            "upperarm_roll_joint": 0.0,
            "elbow_flex_joint": 1.4,
            "forearm_roll_joint": 0.0,
            "wrist_flex_joint": 0.6,
            "wrist_roll_joint": 0.0,
            "l_gripper_finger_joint": 0.03,
            "r_gripper_finger_joint": 0.03,
        }

        for joint_name, angle in crane_pose.items():
            if joint_name not in self.joint_name_to_id:
                continue
            j_global = self.joint_name_to_id[joint_name]

            if j_global in self.joint_indices:
                idx = self.joint_indices.index(j_global)
                low = self.joint_lower[idx]
                high = self.joint_upper[idx]
                init_qpos[idx] = np.clip(angle, low, high)

        print(f"joint_indices:{self.joint_indices} | joint_upper:{self.joint_upper} | joint_lower:{self.joint_lower}")
        self.init_qpos = init_qpos
        self._set_qpos(self.init_qpos)

    #TODO: changes for object id for randomization
    def _load_object_table(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.pb_physics_client)
        self.table_id = p.loadURDF("table/table.urdf", baseOrientation=p.getQuaternionFromEuler([0,0,np.pi/2]), useFixedBase=1, physicsClientId=self.pb_physics_client)
        _, table_aabb_max = p.getAABB(self.table_id, physicsClientId=self.pb_physics_client)
        self.table_z = float(table_aabb_max[2])

        obj_guess_z = self.table_z + 0.15

        # hyperparameters
        ycb_object_dir_path = "./assets/ycb_objects/"
        ycb_objects_paths = sorted(list(Path(ycb_object_dir_path).glob("*")))
        assert 0 <= self.object_idx and self.object_idx < len(ycb_objects_paths), f"object_idx should be in [0, {len(ycb_objects_paths)-1}]"
        object_urdf_path = (ycb_objects_paths[self.object_idx] / "coacd_decomposed_object_one_link.urdf").absolute()
        object_mesh_path = (ycb_objects_paths[self.object_idx] / "textured.obj").absolute()
        self.object_id = p.loadURDF(str(object_urdf_path), basePosition=[0, 0, obj_guess_z], useFixedBase=0, physicsClientId=self.pb_physics_client)
        self.object_canonical_mesh = trimesh.load(str(object_mesh_path))
        object_canonical_pc, face_indices = trimesh.sample.sample_surface(self.object_canonical_mesh, 1024)
        self.object_canonical_pc = object_canonical_pc.astype(np.float32)  # (1024, 3)
        self.object_canonical_normals = self.object_canonical_mesh.face_normals[face_indices].astype(np.float32)  # (1024, 3), outward normals

        # Get object size
        obj_min, obj_max = p.getAABB(self.object_id, physicsClientId=self.pb_physics_client)
        obj_height = obj_max[2] - obj_min[2]
        half_h = 0.5 * obj_height

        #Compute center z so bottom just touches table
        eps = 0.002  # 2mm gap to avoid interpenetration
        obj_center_z = self.table_z + half_h + eps

        #Reset object pose
        x, y = 0.0, 0.0
        p.resetBasePositionAndOrientation(
            self.object_id,
            [x, y, obj_center_z],
            [0, 0, 0, 1],
            physicsClientId=self.pb_physics_client,
        )
    
    def _load_object_goal(self, ):
        self.goal_pos = np.array([self.maze_out_pos_x, self.maze_out_pos_y, 0.9])
        radius = 0.05
        rgba = (0.0, 1.0, 0.0, 0.9)
        vs = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            visualFramePosition=[0, 0, 0],
            physicsClientId=self.pb_physics_client,
        )
        self.goal_marker_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs,
            basePosition=self.goal_pos,
            baseOrientation=[0, 0, 0, 1],
            useMaximalCoordinates=True,
            physicsClientId=self.pb_physics_client,
        )
        p.setCollisionFilterGroupMask(self.goal_marker_id, -1, 0, 0, physicsClientId=self.pb_physics_client)

    def _load_maze(self):
        # maze hyperparameters
        n_rows = 12
        n_cols = 16
        room_width = 6
        maze_offset = 2.0
        grid_xy = 1.0
        grid_z = 2.0
        # need to randomize
        random_row_in = np.random.randint(1, n_rows-1)
        random_row_out = np.random.randint(1, n_rows-1) 
        in_pos = (random_row_in, 0)
        out_pos = (random_row_out, n_cols-1)
        
        self.maze_out_pos_x = n_cols * grid_xy - grid_xy / 2 + maze_offset
        self.maze_out_pos_y = n_rows / 2 * grid_xy - out_pos[0] * grid_xy - grid_xy / 2

        maze = generate_maze_map(
            n_rows=n_rows, n_cols=n_cols,
            in_pos=in_pos, out_pos=out_pos,
            grid_density=0.8,
            add_x_min_wall=True, add_x_max_wall=True,
            add_y_min_wall=True, add_y_max_wall=True,
            p_greedy=0.6,
            seed=5,
        )

        maze = add_left_room_to_maze(maze, room_width=room_width)

        self.cube_positions = create_maze_urdf(
            maze, 
            urdf_path="./assets/maze.urdf",
            grid_xy=grid_xy, grid_z=grid_z,
            maze_center_x=(n_cols-room_width)/2*grid_xy + maze_offset, 
            maze_center_y=0.0, 
            maze_center_z=0.0,
            box_color=(0.95, 0.95, 0.98, 1.0),
        )

        p.loadURDF("./assets/maze.urdf", useFixedBase=1, flags=p.URDF_MERGE_FIXED_LINKS, physicsClientId=self.pb_physics_client)
        
    def visualize_pc_and_normals(self, object_pc, object_normals, visualize_normals=False):
        # remove old debug items
        if self.debug_point_id is not None:
            p.removeUserDebugItem(self.debug_point_id, physicsClientId=self.pb_physics_client)
        for lid in self.debug_line_ids:
            p.removeUserDebugItem(lid, physicsClientId=self.pb_physics_client)
        self.debug_line_ids.clear()

        # add new point cloud
        self.debug_point_id = p.addUserDebugPoints(
            pointPositions=object_pc.tolist(),
            pointColorsRGB=[[0, 0, 1]] * object_pc.shape[0],
            pointSize=2.0,
            lifeTime=0,
            physicsClientId=self.pb_physics_client
        )

        # add normals, this may slow down the simulation
        if visualize_normals:
            normal_scale = 0.02
            for i in range(0, object_pc.shape[0], 50):  # e.g. subsample
                start = object_pc[i]
                end = object_pc[i] + normal_scale * object_normals[i]
                lid = p.addUserDebugLine(start, end, [1, 0, 0], 1.5, 0, physicsClientId=self.pb_physics_client)
                self.debug_line_ids.append(lid)

    def _apply_joint_targets(self, target_qpos):
        """
        target_qpos: np.array of shape (len(self.joint_indices),)
                     joint targets in the same order as self.joint_indices.
        """
        position_gains = [0.3, 0.3, 0.3,  # base (won't change in RL)
                          0.3,  # torso
                          0.3, 0.3,  # head
                          0.3, 0.3,  # shoulder pan & lift
                          0.3, 0.3, 0.3, 0.3, 0.3,  # arm joints
                          0.03, 0.03]  # gripper

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_qpos.tolist(),
            forces=[self.max_force] * len(self.joint_indices),
            positionGains=position_gains,
            physicsClientId=self.pb_physics_client,
        )

    def _apply_action(self, action):
        """
        RL-level action: joint deltas for arm + gripper joints.
        action: np.array shape (len(self.ctrl_joints),), values in [-1, 1].
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        n_ctrl = len(self.ctrl_joints)
        if action.size != n_ctrl:
            raise ValueError(f"Action size {action.size} != controlled joints {n_ctrl}")

        # 1) current joint positions in order of self.joint_indices
        qpos, _, _, _ = self._get_state()  # must align with self.joint_indices
        qpos = np.asarray(qpos, dtype=np.float32)

        # 2) start with current positions as target
        target = qpos.copy()

        # 3) Fixing first 6 joints to init-pos
        for j_id in self.non_ctrl_joints:
            idx = self.joint_indices.index(j_id)
            target[idx] = self.init_qpos[idx]

        # 4) apply deltas to ctrl_joints only
        max_delta = 0.05  # radians per step (tune this!)
        for i, joint_id in enumerate(self.ctrl_joints):
            # find its index in joint_indices
            j_idx = self.joint_indices.index(joint_id)

            q_curr = qpos[j_idx]
            dq = float(np.clip(action[i], -1.0, 1.0)) * max_delta
            q_tgt = q_curr + dq

            # clip to joint limits
            low = self.joint_lower[j_idx]
            high = self.joint_upper[j_idx]
            q_tgt = float(np.clip(q_tgt, low, high))

            target[j_idx] = q_tgt

        # 5) send to PyBullet
        self._apply_joint_targets(target)

    def step(self, action):
        """
        Gym/Gymnasium-style step for SAC with joint-delta control + 11D obs.
        """
        self.step_count += 1

        # 1) prev obs
        prev_obs = self._get_obs_gym()

        # 2) apply RL action
        self._apply_action(action)

        # 3) simulate
        for _ in range(self.substeps):
            p.stepSimulation(physicsClientId=self.pb_physics_client)
            # no time.sleep() during training

        # 4) new obs
        obs = self._get_obs_gym()

        obj_min, obj_max = p.getAABB(self.object_id, physicsClientId=self.pb_physics_client)
        obj_base_z = obj_min[2]  # lowest point of object

        # 5) reward
        reward = compute_grasp_reward_better_closing(prev_obs, obs, table_z=self.table_z, obj_base_z=obj_base_z)

        lifted = obj_base_z > self.table_z + 0.05
        knocked_down = obj_base_z < self.table_z - 0.1

        terminated = bool(lifted or knocked_down)
        truncated = bool((self.step_count >= self.max_steps) and not terminated)

        info = {
            "lifted": lifted,
            "knocked_down": knocked_down,
            "timeout": truncated,
        }

        return obs, reward, terminated, truncated, info
    
    def mini_reset(self):
        """Reset the simulation and reload world/robot. Returns initial observation."""
        self._set_qpos(self.init_qpos)
        p.resetBasePositionAndOrientation(self.object_id, [0, 0, 5.0], [0, 0, 0, 1], physicsClientId=self.pb_physics_client)

        return self._get_obs()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # 1. Let Gym/Gymnasium handle seeding bookkeeping
        super().reset(seed=seed)

        # 2. Reset internal counters
        self.step_count = 0

        # 3. Reset the Bullet world
        p.resetSimulation(physicsClientId=self.pb_physics_client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.pb_physics_client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.pb_physics_client)

        # 4. Rebuild the static scene: plane, table, robot, etc.
        local_rng = random.Random()
        self.object_idx = local_rng.randint(0, 10)
        p.loadURDF("plane.urdf", physicsClientId=self.pb_physics_client)
        self._load_scene()


        #for _ in range(100):  # tune: 50–200 steps
        #    p.stepSimulation(physicsClientId=self.pb_physics_client)

        # 5. Put the robot in a known "home" configuration
        #self._reset_robot() #TODO move robot closer maybe

        # 7. Return the initial observation
        obs = self._get_obs_gym().astype(self.observation_space.dtype)

        info = {}
        return obs, info

    def _reset_robot(self):
        target_pos = self.init_qpos.copy()
        target_pos[0] = 0.5

        print("DEBUG default:", p.getNumJoints(self.robot_id))  # default client
        print("DEBUG actual:", p.getNumJoints(self.robot_id, physicsClientId=self.pb_physics_client))  # correct client

        self._apply_joint_targets(target_pos)
        for _ in range(self.substeps*2):
            p.stepSimulation(physicsClientId=self.pb_physics_client)

    def evaluate(self):
        """
        Evaluate distance between the object position and the goal position.
        """
        _, _, object_pos, _ = self._get_state()
        dist_to_goal = np.linalg.norm(object_pos - self.goal_pos)
        success = dist_to_goal < 0.1

        return {
            "dist_to_goal": dist_to_goal,
            "success": success,
        }
    
    def _get_object_obs(self):
        object_pos, object_xyzw = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.pb_physics_client)
        object_rot = np.array(p.getMatrixFromQuaternion(object_xyzw), physicsClientId=self.pb_physics_client).reshape(3, 3)
        object_pc = (self.object_canonical_pc @ object_rot.T) + np.array(object_pos, dtype=np.float32).reshape(1, 3)  # (1024, 3)
        object_normals = (self.object_canonical_normals @ object_rot.T)  # (1024, 3), outward normals
        return {
            "object_pc": object_pc,
            "object_normals": object_normals,
        }
    
    def _get_maze_obs(self):
        pass
    
    def _get_obs(self):
        qpos, qvel, object_pos, object_xyzw = self._get_state() # （15，）（15,）（3,）（4，）
        object_obs = self._get_object_obs()

        # self.visualize_pc_and_normals(object_pc, object_normals, visualize_normals=False)
        return {
            "qpos": deepcopy(qpos),
            "qvel": deepcopy(qvel),
            "object_pos": deepcopy(object_pos),
            "object_xyzw": deepcopy(object_xyzw),
            "goal_pos": deepcopy(self.goal_pos),
            "object_pc": deepcopy(object_obs["object_pc"]),
            "object_normals": deepcopy(object_obs["object_normals"]),
            "cube_positions": deepcopy(self.cube_positions[:, :2]),
        }

    def _get_obs_gym(self):
        qpos, _, object_pos, object_xyzw = self._get_state()  # （15，）（15,）（3,）（4，）

        object_pos = np.array(object_pos, dtype=np.float32)

        #gripper-axis is the joint before ee
        end_effector_state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True, physicsClientId=self.pb_physics_client)
        end_effector_pos = np.array(end_effector_state[4], dtype=np.float32)
        end_effector_orn = end_effector_state[5]

        # Convert quaternion → Euler to get yaw (around z-axis)
        end_effector_rpy = p.getEulerFromQuaternion(end_effector_orn, physicsClientId=self.pb_physics_client)
        end_effector_yaw = np.float32(end_effector_rpy[2])  # single float

        #relative position
        rel_pos = object_pos - end_effector_pos

        #gap b/w fingers
        grip_width = np.float32(qpos[13] + qpos[14])

        all_states = p.getJointStates(
            self.robot_id,
            self.ctrl_joints,
            physicsClientId=self.pb_physics_client,
        )
        q_ctrl = np.array([s[0] for s in all_states], dtype=np.float32)

        min_xyz, max_xyz = p.getAABB(self.object_id, physicsClientId=self.pb_physics_client)

        min_x, min_y, min_z = min_xyz
        max_x, max_y, max_z = max_xyz
        lx = max_x - min_x
        ly = max_y - min_y
        lz = max_z - min_z
        obj_size = np.array([lx, ly, lz], dtype=np.float32)

        #combine
        observation = np.concatenate([
            end_effector_pos,  # 3
            np.array([end_effector_yaw], np.float32),  # 1
            object_pos,  # 3
            rel_pos,  # 3
            np.array([grip_width], np.float32),  # 1
            q_ctrl, #9
            obj_size #3
        ]).astype(np.float32)

        return observation



    def _get_state(self):
        agent_states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.pb_physics_client)
        qpos = np.array([s[0] for s in agent_states], dtype=np.float32)
        qvel = np.array([s[1] for s in agent_states], dtype=np.float32)
        
        object_pos, object_xyzw = p.getBasePositionAndOrientation(self.object_id, self.pb_physics_client)
        return qpos, qvel, np.array(object_pos), np.array(object_xyzw)

    def _set_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.size != len(self.joint_indices):
            raise ValueError("qpos size mismatch.")
        for idx, q in zip(self.joint_indices, qpos):
            p.resetJointState(self.robot_id, idx, q, physicsClientId=self.pb_physics_client)

    @property
    def action_size(self):
        return len(self.joint_indices)

    @property
    def obs_size(self):
        # obs dict size
        size_str = "\n"
        for k, v in self.obs.items():
            if isinstance(v, np.ndarray):
                size_str += f"{k}: {type(v)}, {v.shape}\n"
            else:
                size_str += f"{k}: {type(v)}\n"
        return size_str

    @property
    def agent_qpos(self):
        qpos, _, _, _ = self._get_state()
        return qpos.copy()

def compute_grasp_reward_v2(
    prev_obs, obs, table_z, obj_base_z,
    dist_weight= 0.2,
    hover_bonus=1,
    grasp_bonus=2.0,
    lift_bonus=10.0,           # ⬅️ make success clearly worth it
    time_penalty=-0.01,
    fail_penalty=-5.0,
    height_weight=10.0,        # ⬅️ shaping for object height progress
    dist_abs_weight = 0.5,
):
    """
    prev_obs, obs: np.array shape (11,)
    table_z: float, *resting* object/table height in world coordinates
             (e.g., init_obj_z you stored at reset)
    Returns: scalar reward (float)
    """

    # -------- Unpack previous & current observations --------
    # Previous
    prev_ee_pos   = prev_obs[0:3]
    prev_obj_pos  = prev_obs[4:7]
    prev_rel_pos  = prev_obs[7:10]
    prev_grip     = prev_obs[10]

    # Current
    ee_pos   = obs[0:3]
    obj_pos  = obs[4:7]
    rel_pos  = obs[7:10]
    grip     = obs[10]

    # l,b,h
    obj_length = obs[20]
    obj_width = obs[21]
    obj_height = obs[22]

    # -------- 1) Distance progress (XY only) --------
    prev_dist_xy = np.linalg.norm(prev_rel_pos[0:2])  # distance in XY plane
    dist_xy      = np.linalg.norm(rel_pos[0:2])

    obj_min_xy = min(obj_length, obj_width)
    r_target = np.clip(0.5 * obj_min_xy, 0.02, 0.05)  # 2–5 cm

    d = max(dist_xy - r_target, 0.0)
    r_dist = dist_weight * np.exp(-100 * d * d)

    # --- 2) Hover / Continuous Z-Alignment Bonus ---
    # Rewards alignment to the ideal grasping Z-position
    target_z = obj_pos[2] + 0.1
    dist_to_target_z = abs(ee_pos[2] - target_z)

    Z_ALIGN_DECAY = 100
    # Gaussian/Exponential Reward: Peaks when EE is at target_z
    r_z_align_cont = hover_bonus * np.exp(-Z_ALIGN_DECAY * dist_to_target_z ** 2)

    # Only apply this high-resolution Z reward if XY is close
    r_hover = r_z_align_cont if dist_xy < 0.05 else 0.0


    # -------- 3) Grasp / contact bonus --------
    # Object close in 3D

    open_threshold = max(obj_length, obj_width) + 0.01  # adding 1cm buffer
    open_threshold = min(open_threshold, 0.1) #gripper
    is_open = grip >= open_threshold
    r_open = 0.0
    if dist_xy < 0.1:
        opening_progress = max(0.0, grip - prev_grip)
        # Reward for the action of opening until the threshold is met
        r_open = 0.1 * opening_progress if not is_open else 0.05

    # "Closing" if grip_width decreased (assuming smaller = more closed)
    object_near_ee = np.linalg.norm(rel_pos) < 0.05  # within 5 cm in 3D


    closing = (prev_grip - grip) > 0.0

    r_grasp = grasp_bonus if (object_near_ee and closing) else 0.0

    # -------- 4) Object height progress shaping --------
    prev_obj_z = prev_obj_pos[2]
    obj_z      = obj_pos[2]

    delta_z = obj_z - prev_obj_z      # positive if object is going up
    r_height = height_weight * delta_z

    # -------- 5) Lift success bonus --------
    # success if object is 5 cm above its resting/table height
    small_lift_clearance = 0.03  # 3cm above table
    big_lift_clearance = 0.06  # 6cm above table
    super_lift_clearance = 0.10
    lifted = obj_base_z - table_z
    r_lift = 0.0
    if lifted >= super_lift_clearance:
        r_lift = lift_bonus * 3
    elif lifted >= big_lift_clearance:
        r_lift = lift_bonus * 2
    elif lifted >= small_lift_clearance:
        r_lift = lift_bonus * 1


    # -------- 6) Knock-down / failure penalty --------
    # If object is significantly below table height, we treat it as knocked off.
    # 10 cm below the reference table_z
    knocked_down = obj_base_z < table_z - 0.1
    r_fail = fail_penalty if knocked_down else 0.0

    # -------- 6) Going under the table --------
    table_margin = 0.005  # small epsilon above table

    ee_below_table = ee_pos[2] < table_z + table_margin
    r_table = -1.0 if ee_below_table else 0.0  # tune weight

    # -------- 8) Time penalty --------
    r_time = time_penalty

    # -------- Total reward --------
    reward = (
        r_dist
        + r_hover
        + r_open
        + r_grasp
        + r_height
        + r_lift
        + r_fail
        + r_table
        + r_time
    )
    return float(reward)

def compute_grasp_reward(
    prev_obs, obs, table_z, obj_base_z,
    dist_weight= 1,
    hover_bonus=1,
    grasp_bonus=2.0,
    lift_bonus=10.0,           # ⬅️ make success clearly worth it
    time_penalty=-0.01,
    fail_penalty=-5.0,
    height_weight=10.0,        # ⬅️ shaping for object height progress
    dist_abs_weight = 0.5,
):
    """
    prev_obs, obs: np.array shape (11,)
    table_z: float, *resting* object/table height in world coordinates
             (e.g., init_obj_z you stored at reset)
    Returns: scalar reward (float)
    """

    # -------- Unpack previous & current observations --------
    # Previous
    prev_ee_pos   = prev_obs[0:3]
    prev_obj_pos  = prev_obs[4:7]
    prev_rel_pos  = prev_obs[7:10]
    prev_grip     = prev_obs[10]

    # Current
    ee_pos   = obs[0:3]
    obj_pos  = obs[4:7]
    rel_pos  = obs[7:10]
    grip     = obs[10]

    # l,b,h
    obj_length = obs[20]
    obj_width = obs[21]
    obj_height = obs[22]

    # -------- 1) Distance progress (XY only) --------
    prev_dist_xy = np.linalg.norm(prev_rel_pos[0:2])  # distance in XY plane
    dist_xy      = np.linalg.norm(rel_pos[0:2])

    # Positive if we moved closer, negative if we moved away
    r_dist_progress = dist_weight * (prev_dist_xy - dist_xy)

    # absolute distance penalty
    r_dist_abs = 0.0

    r_dist_abs = -dist_abs_weight * (dist_xy)

    # total distance term
    r_dist = r_dist_progress + r_dist_abs

    # --- 2) Hover / Continuous Z-Alignment Bonus ---
    # Rewards alignment to the ideal grasping Z-position
    target_z = obj_pos[2] + 0.1
    dist_to_target_z = abs(ee_pos[2] - target_z)

    Z_ALIGN_DECAY = 100
    # Gaussian/Exponential Reward: Peaks when EE is at target_z
    r_z_align_cont = hover_bonus * np.exp(-Z_ALIGN_DECAY * dist_to_target_z ** 2)

    # Only apply this high-resolution Z reward if XY is close
    r_hover = r_z_align_cont if dist_xy < 0.05 else 0.0


    # -------- 3) Grasp / contact bonus --------
    # Object close in 3D

    open_threshold = max(obj_length, obj_width) + 0.01  # adding 1cm buffer
    open_threshold = min(open_threshold, 0.1) #gripper
    is_open = grip >= open_threshold
    r_open = 0.0
    if dist_xy < 0.1:
        opening_progress = max(0.0, grip - prev_grip)
        # Reward for the action of opening until the threshold is met
        r_open = 0.1 * opening_progress if not is_open else 0.05

    # "Closing" if grip_width decreased (assuming smaller = more closed)
    object_near_ee = np.linalg.norm(rel_pos) < 0.05  # within 5 cm in 3D


    closing = (prev_grip - grip) > 0.0

    r_grasp = grasp_bonus if (object_near_ee and closing) else 0.0

    # -------- 4) Object height progress shaping --------
    prev_obj_z = prev_obj_pos[2]
    obj_z      = obj_pos[2]

    delta_z = obj_z - prev_obj_z      # positive if object is going up
    r_height = height_weight * delta_z

    # -------- 5) Lift success bonus --------
    # success if object is 5 cm above its resting/table height
    small_lift_clearance = 0.03  # 3cm above table
    big_lift_clearance = 0.05  # 3cm above table
    lifted = obj_base_z > table_z + small_lift_clearance
    r_lift = lift_bonus if lifted else 0.0


    # -------- 6) Knock-down / failure penalty --------
    # If object is significantly below table height, we treat it as knocked off.
    # 10 cm below the reference table_z
    knocked_down = obj_base_z < table_z - 0.1
    r_fail = fail_penalty if knocked_down else 0.0

    # -------- 6) Going under the table --------
    table_margin = 0.005  # small epsilon above table

    ee_below_table = ee_pos[2] < table_z + table_margin
    r_table = -1.0 if ee_below_table else 0.0  # tune weight

    # -------- 8) Time penalty --------
    r_time = time_penalty

    # -------- Total reward --------
    reward = (
        r_dist
        + r_hover
        + r_open
        + r_grasp
        + r_height
        + r_lift
        + r_fail
        + r_table
        + r_time
    )
    return float(reward)

def compute_grasp_reward_better_closing(
    prev_obs, obs, table_z, obj_base_z,
    dist_weight= 1,
    hover_bonus=1,
    grasp_bonus=5.0,
    grasp_sustain_bonus=0.1,
    lift_bonus=10.0,           # ⬅️ make success clearly worth it
    time_penalty=-0.01,
    fail_penalty=-5.0,
    height_weight=10.0,        # ⬅️ shaping for object height progress
    dist_abs_weight = 0.5,
):
    """
    prev_obs, obs: np.array shape (11,)
    table_z: float, *resting* object/table height in world coordinates
             (e.g., init_obj_z you stored at reset)
    Returns: scalar reward (float)
    """

    # -------- Unpack previous & current observations --------
    # Previous
    prev_ee_pos   = prev_obs[0:3]
    prev_obj_pos  = prev_obs[4:7]
    prev_rel_pos  = prev_obs[7:10]
    prev_grip     = prev_obs[10]

    # Current
    ee_pos   = obs[0:3]
    obj_pos  = obs[4:7]
    rel_pos  = obs[7:10]
    grip     = obs[10]

    # l,b,h
    obj_length = obs[20]
    obj_width = obs[21]
    obj_height = obs[22]

    clearance = obj_base_z - table_z

    # -------- 1) Distance progress (XY only) --------
    prev_dist_xy = np.linalg.norm(prev_rel_pos[0:2])  # distance in XY plane
    dist_xy = np.linalg.norm(rel_pos[0:2])

    # define a "close enough" XY radius based on object width
    obj_extent_xy = max(obj_length, obj_width)
    R_near_xy = 0.5 * obj_extent_xy + 0.05  # half-width + 1cm margin

    r_dist_progress = 0.0
    r_dist_abs = 0.0

    if dist_xy > R_near_xy:
        # only shape distance when we're outside the near ring
        r_dist_progress = dist_weight * (prev_dist_xy - dist_xy)

        # penalise being further than R_near_xy
        gap = dist_xy - R_near_xy
        r_dist_abs = -dist_abs_weight * gap
    else:
        # inside the ring: maybe a tiny centering term, or nothing
        r_dist_abs = -0.1 * dist_abs_weight * dist_xy  # very gentle, or set to 0.0

    # total distance term
    r_dist = r_dist_progress + r_dist_abs

    # --- 2) Hover / Continuous Z-Alignment Bonus ---
    # Rewards alignment to the ideal grasping Z-position
    target_z = obj_pos[2] + 0.01
    dist_to_target_z = abs(ee_pos[2] - target_z)

    Z_ALIGN_DECAY = 100
    # Gaussian/Exponential Reward: Peaks when EE is at target_z
    r_z_align_cont = hover_bonus * np.exp(-Z_ALIGN_DECAY * dist_to_target_z ** 2)

    # Only apply this high-resolution Z reward if XY is close
    r_hover = r_z_align_cont if dist_xy < 0.05 else 0.0


    # -------- 3) Grasp / contact bonus --------
    # Object close in 3D

    open_threshold = max(obj_length, obj_width) + 0.01  # adding 1cm buffer
    open_threshold = min(open_threshold, 0.1)  # gripper
    close_threshold = min(obj_length, obj_width) + 0.001  # adding 1mm buffer

    obj_xy_size = max(obj_length, obj_width)
    # adding dynamic buffer
    if obj_xy_size >= 0.05:
        obj_xy_size += 0.02
    else:
        obj_xy_size += 0.01

    near_xy = dist_xy <= obj_xy_size/2
    good_z = (ee_pos[2] >= obj_pos[2] - obj_height/4) and (ee_pos[2] < obj_pos[2] + obj_height/2)
    r_grasp = 0.0
    if near_xy and good_z:
        closing_progress = max(0.0, prev_grip - grip)
        r_grasp = grasp_bonus * closing_progress

        #maybe only give if lifted beyond ground
        clearance_multiplier = 1
        if grip < open_threshold:
            r_grasp += grasp_sustain_bonus * clearance_multiplier
    else:
        is_open = grip >= open_threshold
        opening_progress = max(0.0, grip - prev_grip)
        # Reward for the action of opening until the threshold is met
        r_grasp = 1 * opening_progress if not is_open else 0.01

    # -------- 4) Object height progress shaping --------
    prev_obj_z = prev_obj_pos[2]
    obj_z      = obj_pos[2]

    delta_z = obj_z - prev_obj_z      # positive if object is going up
    r_height = 0.0
    if delta_z >= 0.00:
        r_height = height_weight * delta_z

    if delta_z <= -0.01: #drop my 1cm or more
        r_height = -1.0 #strict penalty

    # -------- 5) Lift success bonus --------
    # success if object is 5 cm above its resting/table height
    small_lift_clearance = 0.02  # 3cm above table
    big_lift_clearance = 0.04  # 3cm above table
    lifted = obj_base_z > table_z + small_lift_clearance
    r_lift = 0
    if obj_base_z >= table_z + big_lift_clearance + 0.01:
        r_lift = lift_bonus * 2
    elif obj_base_z >= table_z + big_lift_clearance:
        r_lift = lift_bonus
    elif obj_base_z >= table_z + small_lift_clearance:
        r_lift = lift_bonus/2


    # -------- 6) Knock-down / failure penalty --------
    # If object is significantly below table height, we treat it as knocked off.
    # 10 cm below the reference table_z
    knocked_down = obj_base_z < table_z - 0.1
    r_fail = fail_penalty if knocked_down else 0.0

    # -------- 6) Going under the table --------
    table_margin = 0.005  # small epsilon above table

    ee_below_table = ee_pos[2] < table_z + table_margin
    r_table = -1.0 if ee_below_table else 0.0  # tune weight

    # -------- 8) Time penalty --------
    r_time = time_penalty

    # -------- Total reward --------
    reward = (
        r_dist
        + r_hover
        + r_grasp
        + r_height
        + r_lift
        + r_fail
        + r_table
        + r_time
    )
    return float(reward)

def compute_grasp_reward_old(
    prev_obs, obs, table_z, obj_base_z,
    dist_weight= 1,
    hover_bonus=0.2,
    grasp_bonus=2.0,
    lift_bonus=10.0,           # ⬅️ make success clearly worth it
    time_penalty=-0.01,
    fail_penalty=-5.0,
    height_weight=10.0,        # ⬅️ shaping for object height progress
    dist_abs_weight = 0.5,
):
    """
    prev_obs, obs: np.array shape (11,)
    table_z: float, *resting* object/table height in world coordinates
             (e.g., init_obj_z you stored at reset)
    Returns: scalar reward (float)
    """

    # -------- Unpack previous & current observations --------
    # Previous
    prev_ee_pos   = prev_obs[0:3]
    prev_obj_pos  = prev_obs[4:7]
    prev_rel_pos  = prev_obs[7:10]
    prev_grip     = prev_obs[10]

    # Current
    ee_pos   = obs[0:3]
    obj_pos  = obs[4:7]
    rel_pos  = obs[7:10]
    grip     = obs[10]

    # -------- 1) Distance progress (XY only) --------
    prev_dist_xy = np.linalg.norm(prev_rel_pos[0:2])  # distance in XY plane
    dist_xy      = np.linalg.norm(rel_pos[0:2])

    # Positive if we moved closer, negative if we moved away
    r_dist_progress = dist_weight * (prev_dist_xy - dist_xy)

    # absolute distance penalty
    r_dist_abs = -dist_abs_weight * dist_xy

    # total distance term
    r_dist = r_dist_progress + r_dist_abs

    # -------- 2) Hover / alignment bonus --------
    close_xy = dist_xy < 0.03                      # within 3 cm in XY
    above    = ee_pos[2] > obj_pos[2] + 0.02       # at least 2 cm above object
    r_hover  = hover_bonus if (close_xy and above) else 0.0

    # -------- 3) Grasp / contact bonus --------
    # Object close in 3D
    object_near_ee = np.linalg.norm(rel_pos) < 0.05  # within 5 cm in 3D

    # "Closing" if grip_width decreased (assuming smaller = more closed)
    closing = (prev_grip - grip) > 0.0

    r_grasp = grasp_bonus if (object_near_ee and closing) else 0.0

    # -------- 4) Object height progress shaping --------
    prev_obj_z = prev_obj_pos[2]
    obj_z      = obj_pos[2]

    delta_z = obj_z - prev_obj_z      # positive if object is going up
    r_height = height_weight * delta_z

    # -------- 5) Lift success bonus --------
    # success if object is 5 cm above its resting/table height
    small_lift_clearance = 0.03  # 3cm above table
    big_lift_clearance = 0.05  # 3cm above table
    lifted = obj_base_z > table_z + small_lift_clearance
    r_lift = lift_bonus if lifted else 0.0

    # -------- 6) Knock-down / failure penalty --------
    # If object is significantly below table height, we treat it as knocked off.
    # 10 cm below the reference table_z
    knocked_down = obj_base_z < table_z - 0.1
    r_fail = fail_penalty if knocked_down else 0.0

    # -------- 6) Going under the table --------
    table_margin = 0.005  # small epsilon above table

    ee_below_table = ee_pos[2] < table_z + table_margin
    r_table = -1.0 if ee_below_table else 0.0  # tune weight

    # -------- 8) Time penalty --------
    r_time = time_penalty

    # -------- Total reward --------
    reward = (
        r_dist
        + r_hover
        + r_grasp
        + r_height
        + r_lift
        + r_fail
        + r_table
        + r_time
    )
    return float(reward)

if __name__ == "__main__":
    env = PickEnv(gui=True, object_idx=10)
    #env.mini_reset()
    #keyboard_controller = KeyBoardController(env)

    #load the model
    model = SAC.load("./models_sac_grasp_best_phase2/reward_23/best_model_submitable_5objs.zip", env=env)


    #TODO: pick the object
    done = False
    terminated = False
    truncated = False
    ep_reward = 0.0
    steps = 0
    info = {}

    obs = env._get_obs_gym()
    while not done and steps < env.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1

    if done and terminated and info["lifted"]:
        print("Picked")
        # tuck robot arm to minimize space
        qpos, _, _, _ = env._get_state()
        print(f"")
        print(f"Picked q_pos: {qpos}")

        #raise_torso_smooth(env.robot_id, qpos, env.joint_name_to_id, env.joint_indices, target_height= 0.35)
        print("Raised torso")
        tuck_arm_smooth(env.robot_id, qpos, env.joint_name_to_id, env.joint_indices, control_hz=120)
        '''
        tuck_arm_pos = tuck_arm(qpos, env.joint_name_to_id, env.joint_indices)
        env._apply_joint_targets(tuck_arm_pos)  # move the arm
        # TODO : make tuck arm a part of env
        for _ in range(env.substeps*5):
            p.stepSimulation(physicsClientId=env.pb_physics_client)
            time.sleep(1. / 240.)
        '''
    print("Picked and tucked")
    #time.sleep(30)
    #calculate robot footprint
    footprint = path_planner.get_robot_footprint(env.robot_id)
    print(f"footprint: {footprint}")

    #using width and depth for radius
    robot_radius = footprint[1] / 2

    # calculate map
    cell_side_size = 0.2
    table_aabb_min, table_aabb_max = p.getAABB(env.table_id, physicsClientId=env.pb_physics_client)
    pp = path_planner.PathPlanner(env.cube_positions, 22, 12, cell_side_size, robot_radius,
                                  (table_aabb_min, table_aabb_max))
    pp.generate_map()

    # calculate path
    robot_pos, _ = p.getBasePositionAndOrientation(env.robot_id, env.pb_physics_client)
    qpos, _, _, _ = env._get_state()
    # actual robot position is xy-values from the robot base
    robot_x = round(robot_pos[0] + qpos[0], 2)
    robot_y = round(robot_pos[1] + qpos[1], 2)

    path = []
    if env.use_astar:
        path = pp.astar_2d((robot_x, robot_y), (env.goal_pos[0], env.goal_pos[1]))
    else:
        path = pp.dijkstra_2d((robot_x, robot_y), (env.goal_pos[0], env.goal_pos[1]))
    print(f"Path to be taken: {path}")

    pre_movement_pos, _, _, _ = env._get_state()

    #move the robot along the path
    path_idx = 0
    while True:
        # random_action = np.random.uniform(-1.0, 1.0, size=(env.action_size,))
        # action = keyboard_controller.get_action()
        qpos, _, _, _ = env._get_state()

        # Reset the debug visualizer camera
        utils.set_camera_on_robot(qpos[0] - robot_pos[0], qpos[1] - robot_pos[1])

        updated_pos, path_idx, is_complete = pp.follow_path_with_item(path, path_idx, qpos, (robot_pos[0], robot_pos[1]), has_item=True)
        if not is_complete:
            #grasping

            #keep arm locked in-position
            action = pre_movement_pos.copy()
            #updating base positions
            action[0] = updated_pos[0]
            action[1] = updated_pos[1]
            action[2] = updated_pos[2]
            print(f"Updated action: {updated_pos} | action: {action}")
            #gripper
            action[13] = updated_pos[13]
            action[14] = updated_pos[14]

            #move_robot(env.robot_id, qpos, action, env.joint_name_to_id, env.joint_indices, duration=0.01)
            #TODO: keep the grasp also in locked position
            env._apply_joint_targets(action)
            # for k, v in info.items():
            #    print(f"{k}: {v}")
            p.stepSimulation()
            time.sleep(1. / 240.)
        else:
            print("Path complete")
        time.sleep(1. / 240.)


