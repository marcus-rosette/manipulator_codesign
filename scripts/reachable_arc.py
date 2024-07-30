import pybullet as p
import pybullet_data
import numpy as np
import time
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
        # self.prune_point_1 = self.load_urdf("sphere2.urdf", self.prune_point_1_pos, radius=self.radius)
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
        theta = np.linspace(3 * np.pi/2 - allowance_angle, 3 * np.pi/2 + allowance_angle, num_points)

        # Set up arc length coordinate
        x = np.full_like(theta, prune_point[0])  # x-coordinate remains constant
        z = - radius * np.cos(theta) + prune_point[2] # multiply by a negative to mirror on other side of axis
        y = radius * np.sin(theta) + prune_point[1] 

        # Calculate orientation angles
        arc_angles = np.arctan2(prune_point[1] - y, prune_point[2] - z)

        arc_coords = np.vstack((x, y, z))

        goal_coords = np.zeros((num_points, 3)) # 3 for x, y, z
        goal_orientations = np.zeros((num_points, 4)) # 4 for quaternion
        for i in range(num_points):
            goal_coords[i] = [arc_coords[0][i], arc_coords[1][i], arc_coords[2][i]]
            goal_orientations[i] = p.getQuaternionFromEuler([x_ori_default, arc_angles[i], z_ori_default])

        return goal_coords, goal_orientations
        
    def set_joint_positions(self, joint_positions):
        for i in range(self.num_controllable_joints):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, joint_positions[i])
    
    def get_joint_positions(self):
        return [p.getJointState(self.robotId, i)[0] for i in range(self.num_controllable_joints)]
    
    def get_link_state(self, link_idx):
        link_state = p.getLinkState(self.robotId, link_idx)
        link_position = np.array(link_state[0])
        link_orientation = np.array(link_state[1])
        return link_position, link_orientation
    
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
    
    def jacobian_viz(self, jacobian, end_effector_pos):
        # Visualization of the Jacobian columns
        num_columns = jacobian.shape[1]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]  # Different colors for each column
        for i in range(num_columns):
            vector = jacobian[:, i]
            start_point = end_effector_pos
            end_point = start_point + 0.3 * vector[:3]  # Scale the vector for better visualization
            p.addUserDebugLine(start_point, end_point, colors[i % len(colors)], 2)

    def calculate_manipulability(self, joint_positions, planar=True, visualize_jacobian=False):
        zero_vec = [0.0] * len(joint_positions)
        jac_t, jac_r = p.calculateJacobian(self.robotId, self.end_effector_index, [0, 0, 0], joint_positions, zero_vec, zero_vec)
        jacobian = np.vstack((jac_t, jac_r))
        
        if planar:
            jac_t = np.array(jac_t)[1:3]
            jac_r = np.array(jac_r)[0]
            jacobian = np.vstack((jac_t, jac_r))

        if visualize_jacobian:
            end_effector_pos, _ = self.get_link_state(self.end_effector_index)
            self.jacobian_viz(jacobian, end_effector_pos)

        return np.sqrt(np.linalg.det(jacobian @ jacobian.T))
    
    def simple_controller(self, target_joint_positions, position_tol=0.1):
        # Iterate over the joints and set their positions
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robotId, i)
            joint_type = joint_info[2]

            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                p.setJointMotorControl2(
                    bodyIndex=self.robotId,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_joint_positions[i]
                )

        # Wait until the manipulator reaches the target positions
        while True:
            joint_states = [p.getJointState(self.robotId, i)[0] for i in range(self.num_joints)]
            if all(abs(joint_states[i] - target_joint_positions[i]) < position_tol for i in range(len(target_joint_positions))):
                break
            p.stepSimulation()
            time.sleep(0.01)

        # Plot manipulator manipulability after reaching target positions
        manipulability = self.calculate_manipulability(list(target_joint_positions), planar=True, visualize_jacobian=True)
        print(manipulability)

        # Run the simulation for visualization
        p.setRealTimeSimulation(1)

        # Keep the simulation running
        while True:
            p.stepSimulation()
            time.sleep(0.01)

    def vector_field_sample_fn(self, goal_position, goal_orientation, alpha=0.8):
        def sample():
            random_conf = np.random.uniform([limit[0] for limit in self.joint_limits], 
                                            [limit[1] for limit in self.joint_limits])
            self.set_joint_positions(random_conf)
            end_effector_position, _ = self.get_link_state(self.end_effector_index)
            
            vector_to_goal = np.array(goal_position) - end_effector_position
            guided_position = end_effector_position + vector_to_goal
            # guided_conf = np.array(self.inverse_kinematics(guided_position, goal_orientation))
            guided_conf = np.array(p.calculateInverseKinematics(self.robotId, self.end_effector_index, guided_position))
            final_conf = (1 - alpha) * random_conf + alpha * guided_conf
            
            return final_conf
        return sample
    
    def new_sample_fn(self, goal_position, goal_orientation, alpha=0.8):
        def sample():
            def generate_sample():
                # Sample a random configuration within joint limits
                random_conf = np.random.uniform([limit[0] for limit in self.joint_limits], 
                                                [limit[1] for limit in self.joint_limits])
                self.set_joint_positions(random_conf)
                
                # Get the current end-effector position
                end_effector_position, _ = self.get_link_state(self.end_effector_index)
                
                # Calculate the vector towards the goal position
                vector_to_goal = np.array(goal_position) - end_effector_position
                guided_position = end_effector_position + vector_to_goal
                
                # Perform inverse kinematics to find the configuration for the guided position
                guided_conf = np.array(p.calculateInverseKinematics(self.robotId, self.end_effector_index, guided_position))
                
                # Blend the random configuration and guided configuration using alpha
                final_conf = (1 - alpha) * random_conf + alpha * guided_conf
                
                return final_conf
            
            while True:
                final_conf = generate_sample()
                self.set_joint_positions(final_conf)
                end_effector_position, _ = self.get_link_state(self.end_effector_index)
                
                if end_effector_position[1] >= 0:  # Check if y >= 0 (in front of the xz plane)
                    break  # Accept the sample if the end-effector is in front of the xz plane
            
            return final_conf

        return sample

      
    def rrtc_loop(self, goal_coord, goal_ori, start_conf, goal_conf, sample_fn, distance_fn, extend_fn, collision_fn, pos_tol=0.1, ori_tol=0.5, planar=True, max_iter=1000):
        manipulability_max = 0
        path_best = None
        
        for i in range(max_iter):
            path = None
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
                self.set_joint_positions(final_joint_positions)
                final_end_effector_pos, final_end_effector_orientation = self.get_link_state(self.end_effector_index)

                within_tol = self.check_pose_within_tolerance(final_end_effector_pos, final_end_effector_orientation, goal_coord, goal_ori, pos_tol, ori_tol)
                if within_tol:

                    # Calculate and print manipulability
                    manipulability = self.calculate_manipulability(final_joint_positions, planar=planar)
                    if manipulability > manipulability_max:
                        manipulability_max = manipulability
                        path_best = path
                else:
                    path = None

        return manipulability_max, path_best
    

def main():
    planar_pruner = PlanarPruner(urdf_filename="rrpr_manipulator")

    simple_control = False

    prune_point = planar_pruner.prune_point_1_pos

    num_points = 20

    goal_coords, goal_orientations = planar_pruner.prune_arc(prune_point,
                                                              radius=0.1, 
                                                              allowance_angle=np.deg2rad(30), 
                                                              num_points=num_points, 
                                                              x_ori_default=0, 
                                                              z_ori_default=np.pi/2)
    
    if simple_control:
        # poi = int(num_points / 2)
        poi = 10
        target_joint_positions = np.array(p.calculateInverseKinematics(planar_pruner.robotId, planar_pruner.end_effector_index, goal_coords[poi], goal_orientations[poi]))

        length = 0.1
        for i in range(num_points):
            start_point = goal_coords[i]
            # print(start_point)

            # Convert back to euler (SOMETIMES HAS ABIGUITY BETWEEN -PI TO PI)
            angle = p.getEulerFromQuaternion(goal_orientations[i])
            end_point = [start_point[0],
                        start_point[1] + length * np.sin(angle[1]),
                        start_point[2] + length * np.cos(angle[1])
                        ]
            # p.addUserDebugLine(start_point, end_point, (1, 0, 0), 2)

        planar_pruner.simple_controller(target_joint_positions, position_tol=0.001)

    else:
        data_filename = "rrrp_z_arc_manip"

        input(f"\nSaving data to: {data_filename}.csv. Press Enter if correct")

        start_end_effector_pos, start_end_effector_orientation = planar_pruner.get_link_state(planar_pruner.end_effector_index)

        start_conf = planar_pruner.inverse_kinematics(start_end_effector_pos, start_end_effector_orientation)
        controllable_joints = list(range(planar_pruner.num_controllable_joints))
        distance_fn = get_distance_fn(planar_pruner.robotId, controllable_joints)
        extend_fn = get_extend_fn(planar_pruner.robotId, controllable_joints)
        collision_fn = get_collision_fn(planar_pruner.robotId, controllable_joints, planar_pruner.collision_objects)

        sys_manipulability = np.zeros((num_points, 1))

        for point in range(num_points):
            sample_fn = planar_pruner.vector_field_sample_fn(goal_coords[point], goal_orientations[point], alpha=0.6)
            sample_fn = planar_pruner.new_sample_fn(goal_coords[point], goal_orientations[point], alpha=0.6)
            # sample_fn = get_sample_fn(planar_pruner.robotId, controllable_joints)

            goal_conf = planar_pruner.inverse_kinematics(goal_coords[point], goal_orientations[point])
            
            manipulability_max, path_best = planar_pruner.rrtc_loop(goal_coords[point], goal_orientations[point], 
                                                                start_conf, goal_conf, 
                                                                sample_fn, 
                                                                distance_fn, 
                                                                extend_fn, 
                                                                collision_fn, 
                                                                pos_tol=0.1,
                                                                ori_tol=0.5,
                                                                planar=True, 
                                                                max_iter=500)
            
            print(f"Highest manipulability found: {np.round(manipulability_max, 5)}")
            sys_manipulability[point] = manipulability_max

        print("Finished!\n")

        # np.savetxt(f"./data/{data_filename}.csv", np.hstack((goal_coords, goal_orientations, sys_manipulability)))

        p.disconnect()


if __name__ == "__main__":
    main()
