import numpy as np
import time
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from pybullet_planning import (rrt_connect, get_distance_fn, get_sample_fn, get_extend_fn, get_collision_fn)


class LoadRobot:
    def __init__(self, con, robot_urdf_path: str, start_pos, start_orientation, home_config, collision_objects=None) -> None:
        """ Robot loader class

        Args:
            con (class): PyBullet client - an instance of the started env
            robot_urdf_path (str): filename/path to urdf file of robot
            start_pos (float list):  starting origin
            start_orientation (float list): starting orientation as a quaternion
        """
        assert isinstance(robot_urdf_path, str)

        self.con = con
        self.robot_urdf_path = robot_urdf_path
        self.start_pos = start_pos
        self.start_orientation = start_orientation
        self.home_config = home_config
        self.robotId = None
        self.home_ee_pos = None
        self.home_ee_ori = None
        self.collision_objects = collision_objects

        self.setup_robot()

    def setup_robot(self):
        """ Initialize robot
        """
        assert self.robotId is None
        flags = self.con.URDF_USE_SELF_COLLISION

        self.robotId = self.con.loadURDF(self.robot_urdf_path, self.start_pos, self.start_orientation, useFixedBase=True, flags=flags)
        self.num_joints = self.con.getNumJoints(self.robotId)

        self.end_effector_index = self.num_joints - 2
        print(f'\nSelected end-effector index info: {self.con.getJointInfo(self.robotId, self.end_effector_index)[:2]}')

        self.controllable_joint_idx = [
            self.con.getJointInfo(self.robotId, joint)[0]
            for joint in range(self.num_joints)
            if self.con.getJointInfo(self.robotId, joint)[2] in {self.con.JOINT_REVOLUTE, self.con.JOINT_PRISMATIC}
        ]

        # Extract joint limits from urdf
        self.joint_limits = [self.con.getJointInfo(self.robotId, i)[8:10] for i in self.controllable_joint_idx]
        self.lower_limits = [t[0] for t in self.joint_limits]
        self.upper_limits = [t[1] for t in self.joint_limits]
        self.joint_ranges = [upper - lower for lower, upper in zip(self.lower_limits, self.upper_limits)]

        # Set the home position
        self.reset_joint_positions(self.home_config)

        # Get the starting end-effector pos
        self.home_ee_pos, self.home_ee_ori = self.get_link_state(self.end_effector_index)

    def set_joint_positions(self, joint_positions):
        for i, joint_idx in enumerate(self.controllable_joint_idx):
            self.con.setJointMotorControl2(self.robotId, joint_idx, self.con.POSITION_CONTROL, joint_positions[i])

    def reset_joint_positions(self, joint_positions):
        for i, joint_idx in enumerate(self.controllable_joint_idx):
            self.con.resetJointState(self.robotId, joint_idx, joint_positions[i])
            self.con.stepSimulation()

    def set_joint_path(self, joint_path):
        # Vizualize the interpolated positions
        for config in joint_path:
            self.con.setJointMotorControlArray(self.robotId, self.controllable_joint_idx, self.con.POSITION_CONTROL, targetPositions=config)
            self.con.stepSimulation()
            time.sleep(1/25)

    def get_joint_positions(self):
        return [self.con.getJointState(self.robotId, i)[0] for i in self.controllable_joint_idx]
    
    def get_link_state(self, link_idx):
        link_state = self.con.getLinkState(self.robotId, link_idx)
        link_position = np.array(link_state[0])
        link_orientation = np.array(link_state[1])
        return link_position, link_orientation
    
    def check_self_collision(self, joint_config):
        # Set the joint state and step the simulation
        self.reset_joint_positions(joint_config)

        # Return collision bool
        return self.con.getContactPoints(bodyA=self.robotId, bodyB=self.robotId)
    
    def check_collision_aabb(self, robot_id, plane_id):
        # Get AABB for the plane (ground)
        plane_aabb = self.con.getAABB(plane_id)

        # Iterate over each link of the robot
        for i in self.controllable_joint_idx:
            link_aabb = self.con.getAABB(robot_id, i)
            
            # Check for overlap between AABBs
            if (link_aabb[1][0] >= plane_aabb[0][0] and link_aabb[0][0] <= plane_aabb[1][0] and
                link_aabb[1][1] >= plane_aabb[0][1] and link_aabb[0][1] <= plane_aabb[1][1] and
                link_aabb[1][2] >= plane_aabb[0][2] and link_aabb[0][2] <= plane_aabb[1][2]):
                return True

        return False
    
    def inverse_kinematics(self, position, orientation=None, pos_tol=1e-4, rest_config=None):
        if rest_config is None:
            rest_config = self.home_config

        if orientation is not None:
            joint_positions = self.con.calculateInverseKinematics(
                self.robotId, 
                self.end_effector_index, 
                position, 
                orientation, 
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=rest_config,
                residualThreshold=pos_tol
                )
        else:
            joint_positions = self.con.calculateInverseKinematics(
                self.robotId, 
                self.end_effector_index, 
                position, 
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=rest_config,
                residualThreshold=pos_tol
                )
        return joint_positions
    
    def minimize_angle_change(self, start_angle, end_angle):
        """
        Finds the shortest path between start_angle and end_angle, considering
        the wrapping behavior of angles within [-2pi, 2pi].

        Parameters:
        - start_angle: float, the starting joint angle
        - end_angle: float, the desired final joint angle

        Returns:
        - adjusted_end_angle: float, the adjusted end_angle to minimize the movement
        """
        # Normalize the angles to the range [-pi, pi]
        delta = (end_angle - start_angle + np.pi) % (2 * np.pi) - np.pi
        # return start_angle + delta
        return delta

    def shortest_angular_distance(self, start_configuration, end_configuration):
        """
        Calculate the shortest angular distance between start and end joint configurations.

        Parameters:
        - start_configuration: list or numpy array of initial joint angles
        - end_configuration: list or numpy array of target joint angles

        Returns:
        - adjusted_end_configuration: numpy array of end joint angles modified to take the shortest angular distance to the start configuration
        """
        # Ensure inputs are numpy arrays for vectorized operations
        start_configuration = np.array(start_configuration)
        end_configuration = np.array(end_configuration)

        # Normalize both configurations to [-pi, pi]
        start_configuration = (start_configuration + np.pi) % (2 * np.pi) - np.pi
        end_configuration = (end_configuration + np.pi) % (2 * np.pi) - np.pi

        # Calculate the angular difference between start and end configurations
        angular_difference = end_configuration - start_configuration

        # Wrap the angular difference to be within [-pi, pi]
        adjusted_difference = (angular_difference + np.pi) % (2 * np.pi) - np.pi

        # Check both paths: (end - start) and (end - start - 2*pi)
        adjusted_end_minus_2pi = end_configuration - 2 * np.pi
        adjusted_end_plus_2pi = end_configuration + 2 * np.pi

        # Compute all possible angular differences
        diff_original = adjusted_difference
        diff_minus_2pi = (adjusted_end_minus_2pi - start_configuration + np.pi) % (2 * np.pi) - np.pi
        diff_plus_2pi = (adjusted_end_plus_2pi - start_configuration + np.pi) % (2 * np.pi) - np.pi

        # Choose the smallest angular difference for each joint
        shortest_difference = np.where(np.abs(diff_minus_2pi) < np.abs(diff_original), diff_minus_2pi, diff_original)
        shortest_difference = np.where(np.abs(diff_plus_2pi) < np.abs(shortest_difference), diff_plus_2pi, shortest_difference)

        # Adjust the end configuration based on the shortest angular difference
        adjusted_end_configuration = start_configuration + shortest_difference

        return adjusted_end_configuration

    def interpolate_joint_trajectory(self, start_config, end_config, num_steps):
        """
        Interpolates a joint joint trajectory from start_config to end_config

        Parameters:
        - start_config: numpy array of shape (n,), start joint positions
        - end_config: numpy array of shape (n,), end joint positions
        - num_steps: int, number of interpolation steps

        Returns:
        - interpolated_configs: numpy array of shape (num_steps, n), interpolated joint positions
        """
        
        start_config = np.array(start_config)
        end_config = np.array(end_config)

        # Minimize angular rotation of the last two joints
        end_config[4:] = self.shortest_angular_distance(start_config[4:], end_config[4:])

        # Create an array for the interpolated configurations
        interpolated_configs = np.zeros((num_steps, len(start_config)))

        # Loop over each joint to interpolate using minimal angular changes
        for j in range(len(start_config)):
            # Interpolate linearly between the start and adjusted end angles
            for i in range(num_steps):
                interpolated_value = np.linspace(start_config[j], end_config[j], num_steps)[i]
                
                # Ensure the joint value stays within [-2pi, 2pi] after interpolation
                interpolated_value = np.clip(interpolated_value, self.lower_limits[j], self.upper_limits[j])
                
                # Store the interpolated value in the array
                interpolated_configs[i, j] = interpolated_value

        # Check for collisions in the interpolated path
        collision_in_path = any(self.check_self_collision(config) for config in interpolated_configs)

        return interpolated_configs, collision_in_path
    
    def interpolate_joint_trajectory2(self, start_config, end_config, num_steps):
        """
        Interpolates a joint trajectory from start_config to end_config ensuring but ensuring the base joint actuates first

        Parameters:
        - start_config: numpy array of shape (n,), start joint positions
        - end_config: numpy array of shape (n,), end joint positions
        - num_steps: int, number of interpolation steps

        Returns:
        - interpolated_configs: numpy array of shape (num_steps, n), interpolated joint positions
        """
        
        start_config = np.array(start_config)
        end_config = np.array(end_config)

        # Minimize angular rotation of the last two joints
        end_config[4:] = self.shortest_angular_distance(start_config[4:], end_config[4:])

        # Create an array for the interpolated configurations
        interpolated_configs = np.zeros((num_steps, len(start_config)))

        # Calculate the number of steps for the first joint (5%)
        first_joint_steps = int(num_steps * 0.3)
        
        # Interpolation for the first joint
        for i in range(first_joint_steps):
            interpolated_value = np.linspace(start_config[0], end_config[0], first_joint_steps)[i]
            interpolated_value = np.clip(interpolated_value, self.lower_limits[0], self.upper_limits[0])
            interpolated_configs[i, 0] = interpolated_value

        # Set remaining positions for the other joints
        for j in range(1, len(start_config)):
            # Interpolate linearly between the start and adjusted end angles for other joints
            for i in range(num_steps - first_joint_steps):
                interpolated_value = np.linspace(start_config[j], end_config[j], num_steps)[i]
                interpolated_value = np.clip(interpolated_value, self.lower_limits[j], self.upper_limits[j])
                interpolated_configs[i + first_joint_steps, j] = interpolated_value

        # Ensure the first positions of other joints remain the same
        for i in range(first_joint_steps):
            interpolated_configs[i, 1:] = start_config[1:]

        interpolated_configs[first_joint_steps:, 0] = interpolated_configs[first_joint_steps - 1, 0]

        # Check for collisions in the interpolated path
        collision_in_path = any(self.check_self_collision(config) for config in interpolated_configs)

        return interpolated_configs, collision_in_path

    def peck_traj_gen(self, start_config, start_pose, end_config, end_pose, num_steps):
        """ Interpolates a joint trajectory between a start and end joint configuration, but also has the end effectory pass through a point that 
        is between the start and end effector position and orientation, while keeping the depth (y) at the same position as the start position.

        Args:
            start_config (np.array): starting joint coinfiguration
            start_pose (np.array): starting end effector pose [x, y, z, rz, ry, rz, w]
            end_config (np.array): ending joint coinfiguration
            end_pose (np.array): ending end effector pose [x, y, z, rz, ry, rz, w]
            num_steps (int): number of joint configurations in trajectory

        Returns:
            traj (np.array): joint trajectory
            collision (bool): describes any collisions in the trajectory
        """
        start_position = start_pose[:3]
        start_orientation = start_pose[3:]

        end_position = end_pose[:3]
        end_orientation = end_pose[3:]

        mid_position = np.copy(end_position)
        mid_position[1] = start_position[1]

        rotations = R.from_quat([start_orientation, end_orientation])

        # Define key times (e.g., t=0 for start, t=1 for end)
        times = np.array([0, 1])

        # Create SLERP object with two rotations
        slerp = Slerp(times, rotations)

        # Interpolate at t = 0.5 (midpoint)
        mid_rotation = slerp(0.5)

        # Get the quaternion for the mid rotation
        mid_orientation = mid_rotation.as_quat()

        test_rest_ee_pos = (start_position + mid_position) / 2 
        test_rest_config = self.inverse_kinematics(test_rest_ee_pos, mid_orientation, rest_config=list(start_config))
        test_rest_config[4:] = self.shortest_angular_distance(start_config[4:], end_config[4:])

        mid_config = self.inverse_kinematics(mid_position, mid_orientation, rest_config=list(test_rest_config))

        first_half_traj, path_collision1 = self.interpolate_joint_trajectory(start_config, mid_config, int(num_steps/2))
        second_half_traj, path_collision2 = self.interpolate_joint_trajectory(first_half_traj[-1], end_config, int(num_steps/2))

        traj = np.vstack((first_half_traj, second_half_traj))

        if path_collision1 or path_collision2:
            collision = True
        else:
            collision = False

        return traj, collision
    
    def peck_traj_gen2(self, start_config, start_pose, end_config, end_pose, num_steps):
        """ Interpolates a joint trajectory between a start and end joint configuration, but also has the end effectory pass through multiple points 
        that are between the start and end effector position and orientation, while keeping the depth (y) at the same position as the start position.

        Args:
            start_config (np.array): starting joint coinfiguration
            start_pose (np.array): starting end effector pose [x, y, z, rz, ry, rz, w]
            end_config (np.array): ending joint coinfiguration
            end_pose (np.array): ending end effector pose [x, y, z, rz, ry, rz, w]
            num_steps (int): number of joint configurations in trajectory

        Returns:
            traj (np.array): joint trajectory
            collision (bool): describes any collisions in the trajectory
        """
        start_position = start_pose[:3]
        start_orientation = start_pose[3:]

        end_position = end_pose[:3]
        end_orientation = end_pose[3:]

        mid_position = np.copy(end_position)
        mid_position[1] = start_position[1]

        num_midpoints = 3
        mid_positions = np.linspace(start_position, mid_position, num_midpoints + 2)
        # mid_positions = np.linspace(start_position, end_position, num_midpoints + 2)
        mid_positions[:, 1] = start_position[1]

        rotations = R.from_quat([start_orientation, end_orientation])

        # Define key times (e.g., t=0 for start, t=1 for end)
        times = np.array([0, 1])

        # Create SLERP object with two rotations
        slerp = Slerp(times, rotations)

        # Interpolate at t = 0.5 (midpoint)
        mid_rotation0 = slerp(0.25)
        mid_rotation1 = slerp(0.5)
        mid_rotation2 = slerp(0.75)

        # Get the quaternion for the mid rotation
        mid_orientation0 = mid_rotation0.as_quat()
        mid_orientation1 = mid_rotation1.as_quat()
        mid_orientation2 = mid_rotation2.as_quat()

        mid_config0 = self.inverse_kinematics(mid_positions[1], mid_orientation0, rest_config=list(start_config))
        first_traj, path_collision0 = self.interpolate_joint_trajectory(start_config, mid_config0, int(num_steps/6))

        mid_config1 = self.inverse_kinematics(mid_positions[2], mid_orientation1, rest_config=list(first_traj[-1]))
        second_traj, path_collision1 = self.interpolate_joint_trajectory(first_traj[-1], mid_config1, int(num_steps/6))

        mid_config2 = self.inverse_kinematics(mid_positions[3], mid_orientation2, rest_config=list(second_traj[-1]))
        third_traj, path_collision2 = self.interpolate_joint_trajectory(second_traj[-1], mid_config2, int(num_steps/6))

        fourth_traj, path_collision3 = self.interpolate_joint_trajectory(third_traj[-1], end_config, int(num_steps/2))

        traj = np.vstack((first_traj, second_traj, third_traj, fourth_traj))

        if path_collision0 or path_collision1 or path_collision2 or path_collision3:
            collision = True
        else:
            collision = False

        return traj, collision
    
    def task_space_path_interp(self, start_config, start_pose, end_config, end_pose, num_steps):
        """ Interpolates a joint trajectory between a start and end joint configuration, where all intermediate configurations
        are generated in task space while maintaining a desired start and end joint configuration.

        Args:
            start_config (np.array): starting joint coinfiguration
            start_pose (np.array): starting end effector pose [x, y, z, rz, ry, rz, w]
            end_config (np.array): ending joint coinfiguration
            end_pose (np.array): ending end effector pose [x, y, z, rz, ry, rz, w]
            num_steps (int): number of joint configurations in trajectory

        Returns:
            interpolated_configs (np.array): joint trajectory
            collision_in_path (bool): describes any collisions in the trajectory
        """
        start_position = start_pose[:3]
        start_orientation = start_pose[3:]

        end_position = end_pose[:3]
        end_orientation = end_pose[3:]

        rotations = R.from_quat([start_orientation, end_orientation])

        # Define key times (e.g., t=0 for start, t=1 for end)
        times = np.array([0, 1])

        # Create SLERP object with two rotations
        slerp = Slerp(times, rotations)

        # Split the number of steps for each phase
        num_steps_phase1 = num_steps // 2  # First half for x and z translation
        num_steps_phase2 = num_steps - num_steps_phase1  # Second half for y translation

        # Create an empty array for storing interpolated joint configurations
        interpolated_configs = np.zeros((num_steps, len(start_config)))
        interpolated_configs[0] = start_config

        # Phase 1: Interpolate x and z first, keep y constant
        for i in range(1, num_steps_phase1):
            # Interpolate x and z linearly between the start and end pose
            interp_x = np.linspace(start_position[0], end_position[0], num_steps_phase1)[i]
            interp_z = np.linspace(start_position[2], end_position[2], num_steps_phase1)[i]

            # Keep y constant as the start y value
            interp_y = start_position[1]

            intermediate_position = np.array([interp_x, interp_y, interp_z])

            # Interpolate orientation using SLERP
            t = i / num_steps_phase1
            intermediate_orientation = slerp(t).as_quat()  # Get interpolated orientation

            # Create intermediate pose (x, y, z) + maintain original orientation
            intermediate_pose = np.concatenate((intermediate_position, intermediate_orientation))

            # Compute joint configuration for this pose using inverse kinematics
            joint_config = self.inverse_kinematics(intermediate_position, intermediate_orientation, rest_config=list(interpolated_configs[i-1]))
            joint_config = self.shortest_angular_distance(interpolated_configs[i-1], joint_config)

            interpolated_configs[i] = joint_config

        # Phase 2: Interpolate y while keeping x and z fixed at their final values
        for i in range(num_steps_phase2):
            # Interpolate y linearly between the intermediate y and end y
            interp_y = np.linspace(start_position[1], end_position[1], num_steps_phase2)[i]

            # Keep x and z fixed at their final values
            interp_x = end_position[0]
            interp_z = end_position[2]

            # Create final pose (x, y, z) + final orientation
            final_position = np.array([interp_x, interp_y, interp_z])
            
            # Interpolate orientation using SLERP
            t = i / num_steps_phase2
            intermediate_orientation = slerp(0.5 + 0.5 * t).as_quat()  # Interpolate for second phase

            intermediate_pose = np.concatenate((final_position, intermediate_orientation))

            # Compute joint configuration for this pose using inverse kinematics
            joint_config = self.inverse_kinematics(final_position, intermediate_orientation, rest_config=list(interpolated_configs[i-1]))
            joint_config = self.shortest_angular_distance(interpolated_configs[i-1], joint_config)
            interpolated_configs[num_steps_phase1 + i] = joint_config
        
        # Force the last configuration to match end_config
        interpolated_configs[-1] = end_config
        interpolated_configs[-1] = self.shortest_angular_distance(interpolated_configs[-2], end_config)

        # Check for collisions in the interpolated path
        collision_in_path = any(self.check_self_collision(config) for config in interpolated_configs)

        return interpolated_configs, collision_in_path

    def sample_path_to_length(self, path, desired_length):
        """ Takes a joint trajectory path of any length and interpolates to a desired array length

        Args:
            path (float list): joint trajectory
            desired_length (int): desired length of trajectory (number of rows)

        Returns:
            float list: joint trajectory of desired length
        """
        path = np.array(path)
        current_path_len = path.shape[0] # Number of rows
        num_joints = path.shape[1] # Numer of columns

        # Generate new indices for interpolation
        new_indices = np.linspace(0, current_path_len - 1, desired_length)

        # Interpolate each column separately
        return np.array([np.interp(new_indices, np.arange(current_path_len), path[:, i]) for i in range(num_joints)]).T

    def vector_field_sample_fn(self, goal_position, alpha=0.8):
        def sample():
            random_conf = np.random.uniform([limit[0] for limit in self.joint_limits], 
                                            [limit[1] for limit in self.joint_limits])
            self.set_joint_positions(random_conf)
            end_effector_position, _ = self.get_link_state(self.end_effector_index)
            
            vector_to_goal = np.array(goal_position) - end_effector_position
            guided_position = end_effector_position + vector_to_goal
            # guided_conf = np.array(self.robot.inverse_kinematics(guided_position, goal_orientation))
            guided_conf = np.array(self.inverse_kinematics(guided_position))
            final_conf = (1 - alpha) * random_conf + alpha * guided_conf
            
            return final_conf
        return sample

    def rrt_path(self, start_positions, end_positions, target_pos=None, steps=100, rrt_iter=500):
        extend_fn = get_extend_fn(self.robotId, self.controllable_joint_idx)
        collision_fn = get_collision_fn(self.robotId, self.controllable_joint_idx, self.collision_objects)
        distance_fn = get_distance_fn(self.robotId, self.controllable_joint_idx)
        # sample_fn = get_sample_fn(self.robotId, self.controllable_joint_idx)
        sample_fn = self.vector_field_sample_fn(target_pos)

        path = rrt_connect(
            start_positions, end_positions,
            extend_fn=extend_fn,
            collision_fn=collision_fn,
            distance_fn=distance_fn,
            sample_fn=sample_fn,
            max_iterations=rrt_iter
        )
        
        # Ensure the path has exactly `steps` joint configurations
        if path:
            path = self.sample_path_to_length(path, steps)
        
        return path

    def quaternion_angle_difference(self, q1, q2):
        # Compute the quaternion representing the relative rotation
        q1_conjugate = q1 * np.array([1, -1, -1, -1])  # Conjugate of q1
        q_relative = self.con.multiplyTransforms([0, 0, 0], q1_conjugate, [0, 0, 0], q2)[1]
        # The angle of rotation (in radians) is given by the arccos of the w component of the relative quaternion
        angle = 2 * np.arccos(np.clip(q_relative[0], -1.0, 1.0))
        return angle
    
    def check_pose_within_tolerance(self, final_position, final_orientation, target_position, target_orientation, pos_tolerance, ori_tolerance):
        pos_diff = np.linalg.norm(np.array(final_position) - np.array(target_position))
        ori_diff = np.pi - self.quaternion_angle_difference(np.array(target_orientation), np.array(final_orientation))
        return pos_diff <= pos_tolerance and np.abs(ori_diff) <= ori_tolerance
    
    def jacobian_viz(self, jacobian, end_effector_pos):
        # Visualization of the Jacobian columns
        num_columns = jacobian.shape[1]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]  # Different colors for each column
        for i in range(num_columns):
            vector = jacobian[:, i]
            start_point = end_effector_pos
            end_point = start_point + 0.3 * vector[:3]  # Scale the vector for better visualization
            self.con.addUserDebugLine(start_point, end_point, colors[i % len(colors)], 2)

    def calculate_manipulability(self, joint_positions, planar=True, visualize_jacobian=False):
        zero_vec = [0.0] * len(joint_positions)
        jac_t, jac_r = self.con.calculateJacobian(self.robotId, self.end_effector_index, [0, 0, 0], joint_positions, zero_vec, zero_vec)
        jacobian = np.vstack((jac_t, jac_r))
        
        if planar:
            jac_t = np.array(jac_t)[1:3]
            jac_r = np.array(jac_r)[0]
            jacobian = np.vstack((jac_t, jac_r))

        if visualize_jacobian:
            end_effector_pos, _ = self.get_link_state(self.end_effector_index)
            self.jacobian_viz(jacobian, end_effector_pos)

        return np.sqrt(np.linalg.det(jacobian @ jacobian.T))