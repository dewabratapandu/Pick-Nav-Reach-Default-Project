import numpy as np
import pybullet as p

class AntiPodalGrasping:
    def __init__(self, robot_id, body_indices, arm_indices, gripper_indices, max_width=0.1, table_height=0.62):
        self.bot_id = robot_id
        self.max_width = max_width
        self.table_height = table_height
        
        # Joint Indices (From your URDF analysis)
        self.body_indices = body_indices
        self.arm_indices = arm_indices         # e.g., [6, 7, 8, 9, 10, 11, 12]
        self.gripper_indices = gripper_indices # e.g., [13, 14]
        self.base_index = 0
        
        self.ee_index = -1
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            link_name = info[12].decode("utf-8")
            if link_name == "gripper_link":
                self.ee_index = i
                print(f"Found End Effector (Palm) at Index: {self.ee_index}")
                break

        if self.ee_index == -1:
            print("Error: Could not find 'gripper_link'!")
        
        # State Machine Variables
        self.state = "IDLE" 
        self.target_grasp = None # (pos, orn)
        self.timer = 0
        self.gripper_length = self.calculate_dynamic_gripper_length()
        
        # Fixed Actions (Head looks down, Torso stays up)
        self.torso_height = 0.3
        self.head_tilt = 0.6 

    def reset(self):
        self.state = "IDLE"
        self.target_grasp = None
        self.timer = 0
    
    def calculate_dynamic_gripper_length(self):
        """
        Dynamically calculates the distance from the Palm (EE) 
        to the center of the fingertips.
        """        
        # 1. Get Palm Position (World Frame)
        # self.ee_index is the index of 'gripper_link' or 'wrist_roll_link'
        palm_state = p.getLinkState(self.bot_id, self.ee_index)
        palm_pos = np.array(palm_state[0])

        # 2. Get Finger Positions
        # self.gripper_indices usually contains [Left_Finger_Joint, Right_Finger_Joint]
        # In PyBullet, Joint Index == Link Index for that joint.
        f1_pos = np.array(p.getLinkState(self.bot_id, 21)[0]) # right finger
        f2_pos = np.array(p.getLinkState(self.bot_id, 22)[0]) # left finger

        # 3. Calculate Center of Fingertips
        fingertip_center = (f1_pos + f2_pos) / 2.0

        # 4. Calculate Distance (The TCP Offset)
        # This is the base length from wrist to finger connection
        base_length = np.linalg.norm(fingertip_center - palm_pos)
        
        # 5. Add a tiny bit for the physical length of the finger tip itself?
        # Usually, the link origin is at the base of the finger. 
        # You might want to add ~2-3cm to reach the "pads" of the finger.
        # But this removes the guesswork of the main 15cm arm!
        
        # Check if it's too small (e.g. if indices are wrong)
        if base_length < 0.01:
            print("Warning: Dynamic Gripper Length seems too small!")
            return 0.10 # Fallback default
            
        print(f"Dynamic Gripper Length Calculated: {base_length:.4f} meters")
        return base_length

    def _solve_ik(self, target_pos, target_orn, current_joint_angles):
        """Internal helper to get arm angles for a target pose."""
        joint_poses = p.calculateInverseKinematics(
            self.bot_id,
            self.ee_index,
            target_pos,
            target_orn,
            # Use current angles to prevent flipping
            restPoses=current_joint_angles,
            jointDamping=[0.01]*15, # Optional: Helps stability
            maxNumIterations=1000,   # INCREASE THIS (Default is ~20)
            residualThreshold=1e-7  # DECREASE THIS (Require higher precision)
        )
        
        return joint_poses

    def get_action(self, obs):
        """
        Main method to be called every timestep in env.step().
        Returns: np.array of shape (15,)
        """
        # 0. Wait until banana is ready to be picked
        is_not_ready = np.any(obs["object_pos"] == 0)
        if (is_not_ready):
            return np.zeros(15)
        
        # 1. Initialize Action Array (15 DoF)
        action = np.zeros(15)
        
        # Default Gripper: Open (0.05)
        gripper_cmd = 0.05
        
        # Extract current arm state for IK regularization
        full_body_q = obs['qpos'] 
        current_arm_q = [obs['qpos'][i] for i in self.arm_indices]

        # --- STATE MACHINE ---
        
        if self.state == "IDLE":
            self.timer += 1
            if self.timer > 50: # Give it 50 steps (~0.2s)
                self.timer = 0
                self.state = "CALCULATE"

        elif self.state == "CALCULATE":
            # Run the Antipodal Sampling Logic
            points = obs['object_pc']
            normals = obs['object_normals']
            
            # (Insert your generate_best_grasp code here)
            # For brevity, assuming self.generate_best_grasp returns (pos, orn)
            self.target_grasp = self.generate_best_grasp(points, normals)
            
            if self.target_grasp[0] is not None:
                print(f"Grasp Found: {self.target_grasp}!\nMoving to Pre-Grasp...")
                self.state = "APPROACH_PRE"
            else:
                print("No grasp found, retrying...")
                # Stay in CALCULATE or fail
            
            action[2:13] = current_arm_q # Hold

        elif self.state == "APPROACH_PRE":
            # Target: 15cm BEHIND the object
            t_pos, t_orn = self.target_grasp
            
            # Calculate Approach Vector (Local X)
            rot_mat = np.array(p.getMatrixFromQuaternion(t_orn)).reshape(3,3)
            approach_vec = rot_mat[:, 0]
            pre_pos = t_pos - (approach_vec * 0.20)
            pre_pos[2] = pre_pos[2] + 0.5
            p.loadURDF("assets/frame_marker/frame_marker_target.urdf", pre_pos, t_orn, useFixedBase=True, globalScaling=0.25)
            
            # Solve IK
            pose_cmd = self._solve_ik(pre_pos, t_orn, full_body_q)
            action[2:13] = pose_cmd[2:13]
            
            # Check if we are there (Error < 2cm)
            curr_pos = obs['qpos'][2:13] # Rough proxy, better to use FK
            # Ideally use p.getLinkState for error check. 
            # Simple Timer based transition for robustness:
            self.timer += 1
            if self.timer > 50: # Give it 50 steps (~0.2s)
                self.timer = 0
                self.state = "APPROACH_TABLE"
        
        elif self.state == "APPROACH_TABLE":
            # Target: 15cm BEHIND the object
            t_pos, t_orn = self.target_grasp
            
            # Calculate Approach Vector (Local X)
            rot_mat = np.array(p.getMatrixFromQuaternion(t_orn)).reshape(3,3)
            approach_vec = rot_mat[:, 0]
            pre_pos = t_pos - (approach_vec * 0.20)
            pre_pos[2] = pre_pos[2] + 0.5
            
            # Solve IK
            pose_cmd = self._solve_ik(pre_pos, t_orn, full_body_q)
            action[0] = pose_cmd[0]
            action[2:13] = pose_cmd[2:13]
            
            # Check if we are there (Error < 2cm)
            curr_pos = obs['qpos'][2:13] # Rough proxy, better to use FK
            # Ideally use p.getLinkState for error check. 
            # Simple Timer based transition for robustness:
            self.timer += 1
            if self.timer > 50: # Give it 50 steps (~0.2s)
                self.timer = 0
                self.state = "APPROACH_FINAL"
        
        elif self.state == "APPROACH_FINAL":
            t_pos, t_orn = self.target_grasp
            
            # 1. Get the Approach Vector (Red Axis / X-Axis) from the quaternion
            rot_mat = np.array(p.getMatrixFromQuaternion(t_orn)).reshape(3, 3)
            approach_vec = rot_mat[:, 0]  # Column 0 is X (Forward)
            
            # 3. Calculate the Wrist Position
            # We "back up" from the banana center along the approach vector
            wrist_target_pos = t_pos - (approach_vec * self.gripper_length)

            # 4. Visualize for debugging (Draw a line from Banana to Wrist)
            p.loadURDF("assets/frame_marker/frame_marker_target.urdf", wrist_target_pos, t_orn, useFixedBase=True, globalScaling=0.25)
            
            # 5. Send THIS wrist position to IK, not the banana position
            pose_cmd = self._solve_ik(wrist_target_pos, t_orn, full_body_q)
            action[0] = pose_cmd[0]
            action[2:13] = pose_cmd[2:13]
            
            self.timer += 1
            if self.timer > 50:
                self.timer = 0
                self.state = "CLOSE_GRIPPER"

        elif self.state == "CLOSE_GRIPPER":
            # REPEAT the same offset logic here so the arm stays still!
            t_pos, t_orn = self.target_grasp
            
            rot_mat = np.array(p.getMatrixFromQuaternion(t_orn)).reshape(3, 3)
            approach_vec = rot_mat[:, 0]
            wrist_target_pos = t_pos - (approach_vec * self.gripper_length)
            
            pose_cmd = self._solve_ik(wrist_target_pos, t_orn, full_body_q)
            action[0] = pose_cmd[0]
            action[2:13] = pose_cmd[2:13]
            
            # COMMAND GRIPPER CLOSED
            gripper_cmd = 0.0
            
            self.timer += 1
            if self.timer > 40: # Wait for gripper to squeeze
                self.timer = 0
                self.state = "LIFT"

        elif self.state == "LIFT":
            # Target: Grasp Pose + Z offset
            t_pos, t_orn = self.target_grasp
            lift_pos = t_pos + np.array([0, 0, 0.4]) # Up 40cm
            
            pose_cmd = self._solve_ik(lift_pos, t_orn, full_body_q)
            action[0] = pose_cmd[0]
            action[2:13] = pose_cmd[2:13]
            
            # Keep Gripper Closed
            gripper_cmd = 0.0
            
            # End condition
            if obs['object_pos'][2] > (self.table_height + 0.05): # If object is off table
                print("Pick Success!")
                # self.state = "DONE" # Or handle next logic

        # 3. Apply Gripper Command
        action[13] = gripper_cmd
        action[14] = gripper_cmd
        print("state:", self.state, ", action:", action)
        return action

    def generate_best_grasp(self, points, normals, num_samples=1000):
        """
        points: (N, 3) numpy array, point cloud of the object we want to pick
        normals: (N, 3) numpy array, orientation of the point cloud
        Returns: target_pos (3,), target_orn (4,) [quaternion]
        """
        
        # 1. Filter out points that are too low (table collision)
        # Assuming Z is up. Adjust threshold slightly above table height (e.g., +1cm)
        height_threshold = self.table_height + 0.025 
        valid_indices = np.where(points[:, 2] > height_threshold)[0]
        if len(valid_indices) < 2:
            return None, None
            
        points = points[valid_indices]
        normals = normals[valid_indices]
        
        # p.addUserDebugPoints(
        #     pointPositions=points,
        #     pointColorsRGB=[[0, 0, 1]] * points.shape[0],
        #     pointSize=2.0,
        #     lifeTime=0
        # )
        
        best_score = -1.0
        best_grasp = (None, None) # (pos, orn)
        best_points = (None, None)

        # 2. Random Sampling Loop
        for _ in range(num_samples):
            # Pick random point A
            idx_a = np.random.randint(0, len(points))
            p_a = points[idx_a]
            n_a = normals[idx_a]

            # Find candidates for point B (simple distance filter)
            # Vectorized distance calculation
            dists = np.linalg.norm(points - p_a, axis=1)
            
            # Candidates must be within max_width and not the same point
            candidates_idx = np.where((dists < self.max_width) & (dists > 0.005))[0]
            
            if len(candidates_idx) == 0:
                continue

            # Check antipodal constraints for all candidates
            for idx_b in candidates_idx:
                p_b = points[idx_b]
                n_b = normals[idx_b]
                
                # Vector connecting the points
                grasp_vector = p_b - p_a
                grasp_dist = np.linalg.norm(grasp_vector)
                grasp_dir = grasp_vector / grasp_dist
                
                # Condition 1: Normals should be opposite (Dot product near -1)
                normal_alignment = np.dot(n_a, n_b)
                
                # Condition 2: Line connecting points should align with normals
                # We want n_a to align with grasp_dir, and n_b to align with -grasp_dir
                geo_alignment_a = np.dot(n_a, grasp_dir)
                geo_alignment_b = np.dot(n_b, -grasp_dir)
                
                # Total Score Calculation
                # We want normal_alignment to be -1, geo_alignments to be 1
                # Higher score is better.
                # Heuristic: Penalize if normals aren't opposite
                if normal_alignment > -0.5: # If angle < 120 degrees, skip
                    continue
                
                # Simple score: Sum of alignments
                score = (geo_alignment_a + geo_alignment_b) - normal_alignment 
                
                if score > best_score:
                    best_score = score
                    
                    # 3. Calculate Pose
                    # Position is midpoint
                    center_pos = (p_a + p_b) / 2.0
                    
                    # Calculate Rotation Matrix
                    # Fetch Gripper Convention:
                    # x_axis = Approach direction (forward)
                    # y_axis = Closing direction (finger movement)
                    # z_axis = Vertical/Orthogonal
                    
                    y_axis = grasp_dir # Align closing axis with the two points
                    
                    # We need an approach vector (x_axis) perpendicular to y_axis
                    # Try to align approach with global negative Z (top-down) 
                    # or towards robot base. Let's try horizontal approach first.
                    
                    # Temporary Z (up)
                    z_temp = np.array([0, 0, 1])
                    
                    # --- FIX START ---
                    # We want the Red Arrow (Approach) to point roughly DOWN (-Z)
                    # This creates a "Top-Down" grasp which is standard for tables.
                    desired_approach = np.array([0, 0, -1]) 
                    
                    # 1. Calculate Z (Blue) first 
                    # Z must be perpendicular to both Grasp (Y) and Approach (Red)
                    z_axis = np.cross(desired_approach, y_axis)
                    if np.linalg.norm(z_axis) < 0.001: 
                        # Edge case: If grasp is vertical, use X as backup
                        z_axis = np.array([1, 0, 0])
                    z_axis = z_axis / np.linalg.norm(z_axis)
                    
                    # 2. Recalculate X (Red) to be perfectly orthogonal
                    # X = Y cross Z
                    x_axis = np.cross(y_axis, z_axis)
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    # --- FIX END ---
                    
                    # Construct Rotation Matrix (3x3)
                    # [x_axis, y_axis, z_axis] as columns
                    new_y = -z_axis
                    new_z = y_axis
                    rot_matrix = np.column_stack((x_axis, new_y, new_z))
                    
                    # Push it 2cm deeper
                    depth_nudge = 0.0
                    center_pos_deep = center_pos + (x_axis * depth_nudge)
                    
                    # Convert to Quaternion (xyzw)
                    quat = self.matrix_to_quaternion(rot_matrix)
                    
                    # Return the DEEP center, not the surface center
                    best_grasp = (center_pos_deep, quat)
                    best_points = (p_a, p_b)

        contact_points = np.array([best_points[0], best_points[1]])
        p.addUserDebugPoints(
            pointPositions=best_points,
            pointColorsRGB=[[1, 0, 1]] * contact_points.shape[0],
            pointSize=20.0,
            lifeTime=0
        )
        return best_grasp

    def matrix_to_quaternion(self, R):
        # Helper to convert 3x3 rotation matrix to pybullet quaternion (x,y,z,w)
        # Using a robust method (or p.getQuaternionFromEuler if you convert to Euler first)
        # For simplicity in this snippet, let's use a PyBullet trick if available, 
        # or a standard math formula.
        
        # Trace method
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            S = np.sqrt(tr+1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw])
    
    def visualize_grasp_debug(self, grasp_pos, grasp_orn, p1=None, p2=None):
        """
        grasp_pos: the calculated center (from best_grasp)
        grasp_orn: the calculated quaternion (from best_grasp)
        p1, p2: The original contact points (optional, if you want to see them)
        """
        
        # 1. Draw the Target Center (Red Sphere)
        p.addUserDebugText("Grasp Center", grasp_pos, textColorRGB=[1, 0, 0])
        p.addUserDebugLine(grasp_pos, [grasp_pos[0], grasp_pos[1], grasp_pos[2]+0.1], [1,0,0], lineWidth=3)

        # 2. Draw the Approach Vector (Where the wrist points)
        # Convert Quat to Matrix to get direction vectors
        rot_matrix = p.getMatrixFromQuaternion(grasp_orn)
        # Reshape into 3x3 matrix
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        x_axis = rot_matrix[:, 0] # Approach direction (Forward)
        y_axis = rot_matrix[:, 1] # Closing direction (Finger movement)
        z_axis = rot_matrix[:, 2] # Up direction
        
        # Draw the coordinate frame at the grasp center
        # RED line = Approach (X)
        p.addUserDebugLine(grasp_pos, grasp_pos + x_axis * 0.1, [1, 0, 0], lineWidth=4)
        # GREEN line = Closing axis (Y) - This should point towards the contact points!
        p.addUserDebugLine(grasp_pos, grasp_pos + y_axis * 0.1, [0, 1, 0], lineWidth=4)
        # BLUE line = Up (Z)
        p.addUserDebugLine(grasp_pos, grasp_pos + z_axis * 0.1, [0, 0, 1], lineWidth=4)

        # 3. (Optional) Draw the contact points if you have them
        if p1 is not None and p2 is not None:
            p.addUserDebugLine(p1, p2, [0, 1, 0], lineWidth=2) # Green line connecting contacts