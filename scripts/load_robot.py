import numpy as np
import time


class LoadRobot:
    def __init__(self, con, robot_urdf_path: str, start_pos, start_orientation) -> None:
        assert isinstance(robot_urdf_path, str)

        self.con = con
        self.robot_urdf_path = robot_urdf_path
        self.start_pos = start_pos
        self.start_orientation = start_orientation
        self.robotId = None

        self.setup_robot()

    def setup_robot(self):
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
        if orientation is not None:
            joint_positions = self.con.calculateInverseKinematics(self.robotId, self.end_effector_index, position, orientation, residualThreshold=pos_tol)
        else:
            joint_positions = self.con.calculateInverseKinematics(self.robotId, self.end_effector_index, position, residualThreshold=pos_tol)
        return joint_positions
    
    def linear_interp_path(self, end_positions, start_positions=[0.0]*6, steps=100):
        interpolated_joint_angles = [np.linspace(start, end, steps) for start, end in zip(start_positions, end_positions)]
        return [tuple(p) for p in zip(*interpolated_joint_angles)]

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