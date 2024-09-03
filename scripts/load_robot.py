import numpy as np
import time
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Slerp, Rotation
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

        self.end_effector_index = self.num_joints - 3
        print(f'\nSelected end-effector index info: {self.con.getJointInfo(self.robotId, self.end_effector_index)[:2]}')

        self.controllable_joint_idx = [
            self.con.getJointInfo(self.robotId, joint)[0]
            for joint in range(self.num_joints)
            if self.con.getJointInfo(self.robotId, joint)[2] in {self.con.JOINT_REVOLUTE, self.con.JOINT_PRISMATIC}
        ]

        # Extract joint limits from urdf
        self.joint_limits = [self.con.getJointInfo(self.robotId, i)[8:10] for i in self.controllable_joint_idx]

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

    def set_joint_path(self, joint_path):
        # Vizualize the interpolated positions
        for config in joint_path:
            self.con.setJointMotorControlArray(self.robotId, self.controllable_joint_idx, self.con.POSITION_CONTROL, targetPositions=config)
            self.con.stepSimulation()
            time.sleep(1/100)

    def get_joint_positions(self):
        return [self.con.getJointState(self.robotId, i)[0] for i in self.controllable_joint_idx]
    
    def get_link_state(self, link_idx):
        link_state = self.con.getLinkState(self.robotId, link_idx)
        link_position = np.array(link_state[0])
        link_orientation = np.array(link_state[1])
        return link_position, link_orientation
    
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
    
    def inverse_kinematics(self, position, orientation=None, pos_tol=1e-4):
        lower_limits = [t[0] for t in self.joint_limits]
        upper_limits = [t[1] for t in self.joint_limits]
        joint_ranges = [upper - lower for lower, upper in zip(lower_limits, upper_limits)]

        if orientation is not None:
            joint_positions = self.con.calculateInverseKinematics(
                self.robotId, 
                self.end_effector_index, 
                position, 
                orientation, 
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=self.home_config,
                residualThreshold=pos_tol
                )
        else:
            joint_positions = self.con.calculateInverseKinematics(
                self.robotId, 
                self.end_effector_index, 
                position, 
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=self.home_config,
                residualThreshold=pos_tol
                )
        return joint_positions
    
    def clip_joint_vals(self, joint_config):
        joint_config = np.array(joint_config)

        limit = np.pi

        # Add 2pi to values less than -2pi
        joint_config = np.where(joint_config < -limit, joint_config + 2 * np.pi, joint_config)

        # Subtract 2pi from values greater than 2pi
        joint_config = np.where(joint_config > limit, joint_config - 2 * np.pi, joint_config)

        return joint_config
    
    def linear_interp_path(self, start_positions, end_positions, steps=100, limit_joints=True):
        """ Interpolate linear joint positions between a start and end configuration

        Args:
            end_positions (float list): end joint configuration
            start_positions (float list, optional): start joint configuration
            steps (int, optional): number of interpolated positions. Defaults to 100.
            limit_joints (bool, optional): whether or not to clip the joints to the closest revolute equivalent

        Returns:
            list of tuple: interpolated joint positions
        """
        if limit_joints:
            start_positions = self.clip_joint_vals(start_positions)
            end_positions = self.clip_joint_vals(end_positions)

        # Interpolate each joint individually
        interpolated_joint_angles = [np.linspace(start, end, steps) for start, end in zip(start_positions, end_positions)]

        # Extract each joint angle to a combined configuration
        return np.array([tuple(p) for p in zip(*interpolated_joint_angles)])
  
    def cubic_interp_path(self, start_positions, end_positions, steps=100, limit_joints=True):
        """Interpolate joint positions using cubic splines between start and end configurations.

        Args:
            end_positions (float list): end joint configuration
            start_positions (float list, optional): start joint configuration
            steps (int, optional): number of interpolated positions. Defaults to 100.
            limit_joints (bool, optional): whether or not to clip the joints to the closest revolute equivalent

        Returns:
            list of tuple: interpolated joint positions
        """
        num_joints = len(start_positions)
        t = np.linspace(0, 1, steps)

        if limit_joints:
            start_positions = self.clip_joint_vals(start_positions)
            end_positions = self.clip_joint_vals(end_positions)

        interpolated_joint_angles = []
        for i in range(num_joints):
            cs = CubicSpline([0, 1], [start_positions[i], end_positions[i]], bc_type='clamped')
            interpolated_joint_angles.append(cs(t))

        return [tuple(p) for p in zip(*interpolated_joint_angles)]
    
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