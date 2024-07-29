import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from pybullet_planning import (rrt_connect, get_distance_fn, get_sample_fn, get_extend_fn, get_collision_fn)

class PlanarPruner:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        camera_distance = 2
        camera_yaw = 90
        camera_pitch = -10
        camera_target_position = [0, 0.75, 0.75]
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)
        self.planeId = p.loadURDF("plane.urdf")
        self.start_sim()

    def load_urdf(self, urdf_name, start_pos=[0, 0, 0], start_orientation=[0, 0, 0], color=None, fix_base=True, radius=None):
        orientation = p.getQuaternionFromEuler(start_orientation)
        if radius is None:
            objectId = p.loadURDF(urdf_name, start_pos, orientation, useFixedBase=fix_base)
        else:
            objectId = p.loadURDF(urdf_name, start_pos, globalScaling=radius, useFixedBase=fix_base)
            p.changeVisualShape(objectId, -1, rgbaColor=[0, 1, 0, 1]) 
        return objectId

    def start_sim(self):
        start_x = 0.5
        start_y = 1
        self.prune_point_0_pos = [start_x, start_y, 1.55] 
        self.prune_point_1_pos = [start_x, start_y - 0.05, 1.1] 
        self.prune_point_2_pos = [start_x, start_y + 0.05, 0.55] 
        self.radius = 0.05 

        # self.leader_branchId = self.load_urdf("./urdf/leader_branch.urdf", [0, start_y, 1.6/2])
        # self.top_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1.5], [0, np.pi / 2, 0])
        # self.mid_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1], [0, np.pi / 2, 0])
        # self.bottom_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 0.5], [0, np.pi / 2, 0])
        # self.collision_objects = [self.leader_branchId, self.top_branchId, self.mid_branchId, self.bottom_branchId, self.planeId]

        # self.prune_point_0 = self.load_urdf("sphere2.urdf", self.prune_point_0_pos, radius=self.radius)
        # self.prune_point_1 = self.load_urdf("sphere2.urdf", self.prune_point_1_pos, radius=self.radius)
        # self.prune_point_2 = self.load_urdf("sphere2.urdf", self.prune_point_2_pos, radius=self.radius)

        self.robotId = p.loadURDF("./urdf/rprr_manipulator.urdf", [start_x, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robotId)

        # Source the end-effector index
        self.end_effector_index = self.num_joints - 1 # Assuming the end-effector is the last joint

        # Define controllable joint types
        controllable_joint_types = [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]

        # Count controllable joints
        self.num_controllable_joints = sum([1 for i in range(self.num_joints) if p.getJointInfo(self.robotId, i)[2] in controllable_joint_types])

        # Extract joint limits from urdf
        self.joint_limits = [p.getJointInfo(self.robotId, i)[8:10] for i in range(self.num_controllable_joints)]
        
    def set_joint_positions(self, joint_positions):
        for i in range(self.num_controllable_joints):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, joint_positions[i])
    
    def get_joint_positions(self):
        return [p.getJointState(self.robotId, i)[0] for i in range(self.num_controllable_joints)]
    
    def is_collision(self):
        return len(p.getContactPoints(self.robotId)) > 0
    
    def inverse_kinematics(self, position, orientation):
        joint_positions = p.calculateInverseKinematics(self.robotId, self.end_effector_index, position, orientation)
        return joint_positions
    
    def quaternion_angle_difference(self, q1, q2):
        # Compute the quaternion representing the relative rotation
        q1_conjugate = q1 * np.array([1, -1, -1, -1])  # Conjugate of q1
        q_relative = p.multiplyTransforms([0, 0, 0], q1_conjugate, [0, 0, 0], q2)[1]
        # The angle of rotation (in radians) is given by the arccos of the w component of the relative quaternion
        angle = 2 * np.arccos(np.clip(q_relative[0], -1.0, 1.0))
        return angle
    
    def check_pose_within_tolerance(self, final_position, final_orientation, target_position, target_orientation, pos_tolerance, ori_tolerance):
        pos_diff = np.linalg.norm(np.array(final_position) - np.array(target_position))
        # ori_diff = np.linalg.norm(np.array(final_orientation) - np.array(target_orientation))
        ori_diff = np.pi - self.quaternion_angle_difference(np.array(target_orientation), np.array(final_orientation))
        # print(pos_diff, ori_diff)
        return (pos_diff <= pos_tolerance and np.abs(ori_diff) <= ori_tolerance, pos_diff, ori_diff)

    def calculate_manipulability(self, joint_positions, planar=True):
        joint_positions = list(joint_positions)
        zero_vec = [0.0] * len(joint_positions)
        jac_t, jac_r = p.calculateJacobian(self.robotId, self.end_effector_index, [0, 0, 0], joint_positions, zero_vec, zero_vec)
        jacobian = np.vstack((jac_t, jac_r))
        
        if planar:
            jac_t = np.array(jac_t)[1:3]
            jac_r = np.array(jac_r)[0]
            jacobian = np.vstack((jac_t, jac_r))

        return np.sqrt(np.linalg.det(jacobian @ jacobian.T))

    def vector_field_sample_fn(self, goal_position, goal_orientation, alpha=0.8):
        def sample():
            random_conf = np.random.uniform([limit[0] for limit in self.joint_limits], 
                                            [limit[1] for limit in self.joint_limits])
            self.set_joint_positions(random_conf)
            end_effector_state = p.getLinkState(self.robotId, self.end_effector_index)
            end_effector_position = np.array(end_effector_state[0])
            
            vector_to_goal = np.array(goal_position) - end_effector_position
            guided_position = end_effector_position + vector_to_goal
            guided_conf = np.array(self.inverse_kinematics(guided_position, goal_orientation))
            final_conf = (1 - alpha) * random_conf + alpha * guided_conf
            
            return final_conf
        return sample

def main():
    planar_pruner = PlanarPruner()
    goal_position = planar_pruner.prune_point_1_pos
    # goal_position = [0.5, 0.6, 0.6]

    # [-0.1830127  0.1830127  0.6830127  0.6830127]
    # [-0.14298942  0.14298942  0.69249839  0.69249839]
    # [-0.1024823   0.1024823   0.69964089  0.69964089]
    # [-0.06162842  0.06162842  0.70441603  0.70441603]
    # [-0.020566    0.020566    0.70680764  0.70680764]
    # [ 0.020566   -0.020566    0.70680764  0.70680764]
    # [ 0.06162842 -0.06162842  0.70441603  0.70441603]
    # [ 0.1024823  -0.1024823   0.69964089  0.69964089]
    # [ 0.14298942 -0.14298942  0.69249839  0.69249839]
    # [ 0.1830127 -0.1830127  0.6830127  0.6830127]

    goal_orientation = p.getQuaternionFromEuler([0, 1.57, 1.57])
    # goal_orientation = [-0.06162842,  0.06162842,  0.70441603,  0.70441603]
    pos_tolerance = 0.1
    ori_tolerance = 0.5

    start_end_effector_state = p.getLinkState(planar_pruner.robotId, planar_pruner.end_effector_index)
    start_end_effector_pos = np.array(start_end_effector_state[0])
    start_end_effector_orientation = np.array(start_end_effector_state[1])

    start_conf = planar_pruner.inverse_kinematics(start_end_effector_pos, start_end_effector_orientation)
    controllable_joints = list(range(planar_pruner.num_controllable_joints))
    distance_fn = get_distance_fn(planar_pruner.robotId, controllable_joints)
    sample_fn = planar_pruner.vector_field_sample_fn(goal_position, goal_orientation, alpha=0.7)
    # sample_fn = get_sample_fn(planar_pruner.robotId, controllable_joints)
    extend_fn = get_extend_fn(planar_pruner.robotId, controllable_joints)
    collision_fn = get_collision_fn(planar_pruner.robotId, controllable_joints, [])#planar_pruner.collision_objects)

    max_iterations = 5000
    num_feasible_path = 0
    manipulability_max = 0
    path = None

    goal_conf = planar_pruner.inverse_kinematics(goal_position, goal_orientation)
    for i in range(max_iterations):
        path = rrt_connect(
            start_conf, goal_conf,
            extend_fn=extend_fn,
            collision_fn=collision_fn,
            distance_fn=distance_fn,
            sample_fn=sample_fn,
            max_iterations=1000
        )

        if path is not None:
            final_joint_positions = path[-1]
            planar_pruner.set_joint_positions(final_joint_positions)
            final_end_effector_state = p.getLinkState(planar_pruner.robotId, planar_pruner.end_effector_index)
            final_end_effector_pos = np.array(final_end_effector_state[0])
            final_end_effector_orientation = np.array(final_end_effector_state[1])

            # print(f'\nTarget position: {goal_position}, Target orientation: {p.getEulerFromQuaternion(goal_orientation)}')
            # print(f'Final position: {np.round(final_end_effector_pos, 2)}, Final orientation: {np.round(p.getEulerFromQuaternion(final_end_effector_orientation), 2)}')
            # print(f"Position error: {np.round(pos_diff, 2)}, Orientation error: {np.round(ori_diff, 2)}")

            within_tol, pos_diff, ori_diff = planar_pruner.check_pose_within_tolerance(final_end_effector_pos, final_end_effector_orientation, goal_position, goal_orientation, pos_tolerance, ori_tolerance)
            if within_tol:
                iteration = i
                print(iteration)
                num_feasible_path += 1
                # print(f'\nTarget position: {goal_position}, Target orientation: {p.getEulerFromQuaternion(goal_orientation)}')
                joint_states = [p.getJointState(planar_pruner.robotId, i)[0] for i in range(planar_pruner.num_joints)]
                print("Current joint configuration:", joint_states)
                print(f'Final position: {np.round(final_end_effector_pos, 2)}, Final orientation: {np.round(p.getEulerFromQuaternion(final_end_effector_orientation), 2)}')
                # print(f"Position error: {np.round(pos_diff, 2)}, Orientation error: {np.round(ori_diff, 2)}")

                # Calculate and print manipulability
                manipulability = planar_pruner.calculate_manipulability(final_joint_positions, planar=True)
                if manipulability > manipulability_max:
                    manipulability_max = manipulability
                print(f'Manipulability at final position: {np.round(manipulability, 5)}')
                # print("")
                break
            else:
                path = None

    if path is None:
        print("No path found within the specified tolerances!")
        print(f"Number of feasible paths found: {num_feasible_path}")
        print(f"Highest manipulability found: {manipulability_max}")
    else:
        print(f"Path found within tolerances on iteration {iteration}! Executing path...")
        print(f"Number of feasible paths found: {num_feasible_path}")
        print(f"Highest manipulability found: {manipulability_max}")

        for config in path:
            planar_pruner.set_joint_positions(config)
            p.stepSimulation()
            time.sleep(0.1)
        print("Path execution complete. Robot should stay at the goal position.")
        
        final_position = path[-1]
        while True:
            planar_pruner.set_joint_positions(final_position)
            p.stepSimulation()
            time.sleep(0.1)

    p.disconnect()

if __name__ == "__main__":
    main()
