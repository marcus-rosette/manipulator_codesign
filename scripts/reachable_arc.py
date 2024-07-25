import pybullet as p
import pybullet_data
import numpy as np
from pybullet_planning import (rrt_connect, get_distance_fn, get_sample_fn, get_extend_fn, get_collision_fn)


class PlanarPruner:
    def __init__(self, urdf_filename="rrr_manipulator"):
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        camera_distance = 2
        camera_yaw = 90
        camera_pitch = -10
        camera_target_position = [0, 0.75, 0.75]
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

        self.urdf_filename = urdf_filename
        self.load_objects()

    def load_urdf(self, urdf_name, start_pos=[0, 0, 0], start_orientation=[0, 0, 0], fix_base=True, radius=None):
        orientation = p.getQuaternionFromEuler(start_orientation)
        if radius is None:
            objectId = p.loadURDF(urdf_name, start_pos, orientation, useFixedBase=fix_base)
        else:
            objectId = p.loadURDF(urdf_name, start_pos, globalScaling=radius, useFixedBase=fix_base)
            p.changeVisualShape(objectId, -1, rgbaColor=[0, 1, 0, 1]) 
        return objectId

    def load_objects(self):
        self.planeId = p.loadURDF("plane.urdf")

        start_x = 0.5
        start_y = 1.0
        self.prune_point_0_pos = [start_x, start_y, 1.55] 
        self.prune_point_1_pos = [start_x, start_y - 0.05, 1.1] 
        self.prune_point_2_pos = [start_x, start_y + 0.05, 0.55] 
        self.radius = 0.05 

        self.leader_branchId = self.load_urdf("./urdf/leader_branch.urdf", [0, start_y, 1.6/2])
        self.top_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1.5], [0, np.pi / 2, 0])
        self.mid_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1], [0, np.pi / 2, 0])
        self.bottom_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 0.5], [0, np.pi / 2, 0])
        self.collision_objects = [self.leader_branchId, self.top_branchId, self.mid_branchId, self.bottom_branchId, self.planeId]

        self.prune_point_0 = self.load_urdf("sphere2.urdf", self.prune_point_0_pos, radius=self.radius)
        self.prune_point_1 = self.load_urdf("sphere2.urdf", self.prune_point_1_pos, radius=self.radius)
        self.prune_point_2 = self.load_urdf("sphere2.urdf", self.prune_point_2_pos, radius=self.radius)

        self.robotId = p.loadURDF(f"./urdf/{self.urdf_filename}.urdf", [start_x, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robotId)

        # Source the end-effector index
        self.end_effector_index = self.num_joints - 1 # Assuming the end-effector is the last joint

        # Define controllable joint types
        controllable_joint_types = [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]

        # Count controllable joints
        self.num_controllable_joints = sum([1 for i in range(self.num_joints) if p.getJointInfo(self.robotId, i)[2] in controllable_joint_types])

        # Extract joint limits from urdf
        self.joint_limits = [p.getJointInfo(self.robotId, i)[8:10] for i in range(self.num_controllable_joints)]

    def prune_arc(self, prune_point, radius, allowance_angle, num_points, x_ori_default=0.0, z_ori_default=np.pi/2):
        # Define theta as a descrete array
        theta = np.linspace(-allowance_angle, allowance_angle, num_points)

        # Set up arc length coordinate
        x = np.full_like(theta, prune_point[0])  # x-coordinate remains constant
        y = - radius * np.cos(theta) + prune_point[1] # multiply by a negative to mirror on other side of axis
        z = radius * np.sin(theta) + prune_point[2] 

        # Calculate orientation angles
        arc_angles = np.arctan2(prune_point[1] - z, prune_point[0] - y)

        arc_coords = np.vstack((x, y, z))

        goal_coords = np.zeros((num_points, 3)) # 3 for x, y, z
        goal_orientations = np.zeros((num_points, 4)) # 4 for quaternion
        for i in range(num_points):
            goal_coords[i] = [arc_coords[0][i], arc_coords[1][i], arc_coords[2][i]]
            goal_orientations[i] = p.getQuaternionFromEuler([x_ori_default, arc_angles[i] + np.pi + 1.57, z_ori_default])

        return goal_coords, goal_orientations
        
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
        ori_diff = np.pi - self.quaternion_angle_difference(np.array(target_orientation), np.array(final_orientation))
        return pos_diff <= pos_tolerance and np.abs(ori_diff) <= ori_tolerance

    def calculate_manipulability(self, joint_positions, planar=True):
        zero_vec = [0.0] * len(joint_positions)
        jac_t, jac_r = p.calculateJacobian(self.robotId, self.end_effector_index, [0, 0, 0], joint_positions, zero_vec, zero_vec)
        jacobian = np.vstack((jac_t, jac_r))
        
        # Check for singularity
        if np.linalg.matrix_rank(jacobian) < self.num_controllable_joints:
            print("\nSingularity detected: Manipulability is zero.")
            return 0.0
        
        if planar:
            jac_t = np.array(jac_t)[1:3]
            manipulability_measure = np.sqrt(np.linalg.det(jac_t @ jac_t.T))
        else:
            manipulability_measure = np.sqrt(np.linalg.det(jacobian @ jacobian.T))
        return manipulability_measure

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
    
    def rrtc_loop(self, goal_coord, goal_ori, start_conf, goal_conf, sample_fn, distance_fn, extend_fn, collision_fn, pos_tol=0.1, ori_tol=0.5, planar=True, max_iter=1000):
        manipulability_max = 0
        path = None
        
        for i in range(max_iter):
            path = rrt_connect(
                start_conf, goal_conf,
                extend_fn=extend_fn,
                collision_fn=collision_fn,
                distance_fn=distance_fn,
                sample_fn=sample_fn,
                max_iterations=10000
            )

            if path is not None:
                final_joint_positions = path[-1]
                self.set_joint_positions(final_joint_positions)
                final_end_effector_state = p.getLinkState(self.robotId, self.end_effector_index)
                final_end_effector_pos = np.array(final_end_effector_state[0])
                final_end_effector_orientation = np.array(final_end_effector_state[1])

                within_tol = self.check_pose_within_tolerance(final_end_effector_pos, final_end_effector_orientation, goal_coord, goal_ori, pos_tol, ori_tol)
                if within_tol:

                    # Calculate and print manipulability
                    manipulability = self.calculate_manipulability(final_joint_positions, planar=planar)
                    if manipulability > manipulability_max:
                        manipulability_max = manipulability
                else:
                    path = None

        if path is None:
            return manipulability_max
    

def main():
    planar_pruner = PlanarPruner(urdf_filename="prrr_manipulator")
    prune_point = planar_pruner.prune_point_1_pos

    num_points = 10

    goal_coords, goal_orientations = planar_pruner.prune_arc(prune_point,
                                                              radius=0.1, 
                                                              allowance_angle=np.deg2rad(30), 
                                                              num_points=num_points, 
                                                              x_ori_default=0, 
                                                              z_ori_default=np.pi/2)

    start_end_effector_state = p.getLinkState(planar_pruner.robotId, planar_pruner.end_effector_index)
    start_end_effector_pos = np.array(start_end_effector_state[0])
    start_end_effector_orientation = np.array(start_end_effector_state[1])

    start_conf = planar_pruner.inverse_kinematics(start_end_effector_pos, start_end_effector_orientation)
    controllable_joints = list(range(planar_pruner.num_controllable_joints))
    distance_fn = get_distance_fn(planar_pruner.robotId, controllable_joints)
    extend_fn = get_extend_fn(planar_pruner.robotId, controllable_joints)
    collision_fn = get_collision_fn(planar_pruner.robotId, controllable_joints, planar_pruner.collision_objects)

    sys_manipulability = np.zeros((num_points, 1))

    for point in range(num_points):
        # sample_fn = planar_pruner.vector_field_sample_fn(goal_coords[point], goal_orientations[point], alpha=0.5)
        sample_fn = get_sample_fn(planar_pruner.robotId, controllable_joints)

        goal_conf = planar_pruner.inverse_kinematics(goal_coords[point], goal_orientations[point])
        
        manipulability_max = planar_pruner.rrtc_loop(goal_coords[point], goal_orientations[point], 
                                                            start_conf, goal_conf, 
                                                            sample_fn, 
                                                            distance_fn, 
                                                            extend_fn, 
                                                            collision_fn, 
                                                            pos_tol=0.1,
                                                            ori_tol=0.5,
                                                            planar=True, 
                                                            max_iter=250)
        
        print(f"Highest manipulability found: {np.round(manipulability_max, 5)}")
        sys_manipulability[point] = manipulability_max

    print("Finished!\n")

    # np.savetxt("./data/arc_manipulability_gripper.csv", np.hstack((goal_coords, goal_orientations, sys_manipulability)))

    p.disconnect()

if __name__ == "__main__":
    main()
