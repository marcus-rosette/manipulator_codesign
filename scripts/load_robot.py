import numpy as np
import time
from scipy.interpolate import CubicSpline


class LoadRobot:
    def __init__(self, con, robot_urdf_path: str, start_pos, start_orientation, home_config) -> None:
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
    
    def is_collision(self):
        return len(self.con.getContactPoints(self.robotId)) > 0
    
    def inverse_kinematics(self, position, orientation=None, pos_tol=1e-4):
        lower_limits = [t[0] for t in self.joint_limits]
        upper_limits = [t[1] for t in self.joint_limits]

        if orientation is not None:
            joint_positions = self.con.calculateInverseKinematics(
                self.robotId, 
                self.end_effector_index, 
                position, 
                orientation, 
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=[upper - lower for lower, upper in zip(lower_limits, upper_limits)],
                # restPoses=[0] * len(self.controllable_joint_idx),
                restPoses=self.home_config,
                residualThreshold=pos_tol)
        else:
            joint_positions = self.con.calculateInverseKinematics(
                self.robotId, 
                self.end_effector_index, 
                position, 
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=[upper - lower for lower, upper in zip(lower_limits, upper_limits)],
                # restPoses=[0] * len(self.controllable_joint_idx),
                restPoses=self.home_config,
                residualThreshold=pos_tol)
        return joint_positions
    
    def linear_interp_path(self, start_positions, end_positions, steps=100):
        """ Interpolate linear joint positions between a start and end configuration

        Args:
            end_positions (float list): end joint configuration
            start_positions (float list, optional): start joint configuration. Defaults to [0.0]*6.
            steps (int, optional): number of interpolated positions. Defaults to 100.

        Returns:
            list: interpolated path
        """
        interpolated_joint_angles = [np.linspace(start, end, steps) for start, end in zip(start_positions, end_positions)]
        return [tuple(p) for p in zip(*interpolated_joint_angles)]

    def spherical_linear_interp_path(self, start_positions, end_positions, steps=100):
        """ Interpolate linear joint positions between a start and end configuration

        Args:
            end_positions (float list): end joint configuration
            start_positions (float list, optional): start joint configuration. Defaults to [0.0]*6.
            steps (int, optional): number of interpolated positions. Defaults to 100.

        Returns:
            list: interpolated path
        """
        interpolated_joint_angles = []
        for start, end in zip(start_positions, end_positions):
            # Wrap angles to the range [-pi, pi]
            start = self.wrap_angle(start)
            end = self.wrap_angle(end)
            
            # Calculate linear interpolation
            delta = end - start
            if abs(delta) > np.pi:
                # Adjust for wrap-around
                if delta > 0:
                    end -= 2 * np.pi
                else:
                    end += 2 * np.pi
                delta = end - start
            
            interpolated_joint_angles.append(np.linspace(start, end, steps))
        
        return [tuple(p) for p in zip(*interpolated_joint_angles)]

    def wrap_angle(self, angle):
        """Wrap angle to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def cubic_interp_path(self, start_positions, end_positions, steps=100):
        """Interpolate joint positions using cubic splines between start and end configurations.

        Args:
            start_positions (list of float): start joint configuration
            end_positions (list of float): end joint configuration
            steps (int, optional): number of interpolated positions. Defaults to 100.

        Returns:
            list of tuple: interpolated joint positions
        """
        num_joints = len(start_positions)
        t = np.linspace(0, 1, steps)

        interpolated_joint_angles = []
        for i in range(num_joints):
            cs = CubicSpline([0, 1], [start_positions[i], end_positions[i]], bc_type='clamped')
            interpolated_joint_angles.append(cs(t))

        return [tuple(p) for p in zip(*interpolated_joint_angles)]
    
    def linear_interp_end_effector_path(self, start_positions, end_positions, steps=100):
        """
        Interpolates a joint trajectory between start_positions and end_positions for a UR5 manipulator in PyBullet.

        Parameters:
        - start_positions: List or array of joint angles for the starting configuration.
        - end_positions: List or array of joint angles for the ending configuration.
        - steps: Number of interpolation steps.

        Returns:
        - trajectory: List of joint configurations representing the interpolated trajectory.
        """
        trajectory = [start_positions]  # Ensure the first config is always the start_config
        end_effector_positions = [self.get_link_state(self.end_effector_index)[0]]  # Get the initial end-effector position

        # Generate linearly interpolated joint angles
        for i in range(steps-1):
            t = i / (steps - 1)
            interpolated_config = np.array(start_positions) * (1 - t) + np.array(end_positions) * t
            trajectory.append(interpolated_config.tolist())
        
        # Ensure linear end-effector path
        for config in trajectory:
            self.reset_joint_positions(config)
            end_effector_pos, _ = self.get_link_state(self.end_effector_index)
            end_effector_positions.append(end_effector_pos)
        
        # Re-interpolate to ensure a linear end-effector path
        end_effector_positions = np.array(end_effector_positions)
        start_pos, end_pos = end_effector_positions[0], end_effector_positions[-1]
        for i in range(steps-1):
            t = i / (steps - 1)
            expected_pos = start_pos * (1 - t) + end_pos * t
            actual_pos = end_effector_positions[i]
            correction = expected_pos - actual_pos

            # Apply correction to the trajectory
            corrected_config = self.inverse_kinematics(actual_pos + correction)
            trajectory[i] = np.array(corrected_config).tolist()
        
        return trajectory
    
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