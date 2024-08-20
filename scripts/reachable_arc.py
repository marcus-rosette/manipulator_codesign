import pybullet as p
import pybullet_data
import numpy as np
import time
from align_ee_hemisphere import sample_hemisphere_suface_pts, end_effector_orientations
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
        start_y = 1
        self.prune_point_0_pos = [start_x, start_y, 1.55] 
        self.prune_point_1_pos = [start_x, start_y - 0.05, 1.1] 
        self.prune_point_2_pos = [start_x, start_y + 0.05, 0.55] 
        self.radius = 0.05 

        self.leader_branchId = self.load_urdf("./urdf/leader_branch.urdf", [0, start_y, 1.6/2])
        self.top_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1.5], [0, np.pi / 2, 0])
        self.mid_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1], [0, np.pi / 2, 0])
        self.bottom_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 0.5], [0, np.pi / 2, 0])
        self.collision_objects = [self.leader_branchId, self.top_branchId, self.mid_branchId, self.bottom_branchId, self.planeId]

        # self.prune_point_0 = self.load_urdf("sphere2.urdf", self.prune_point_0_pos, radius=self.radius)
        self.prune_point_1 = self.load_urdf("sphere2.urdf", self.prune_point_1_pos, radius=self.radius)
        # self.prune_point_2 = self.load_urdf("sphere2.urdf", self.prune_point_2_pos, radius=self.radius)

        self.robotId = p.loadURDF(f"./urdf/{self.urdf_filename}.urdf", [start_x, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robotId)

        # Source the end-effector index
        print('end-effector index for ur5e is num_joints - 2 (instead of - 1)')
        self.end_effector_index = self.num_joints - 2 # Assuming the end-effector is the last joint

        self.controllable_joint_idx = [
            p.getJointInfo(self.robotId, joint)[0]
            for joint in range(self.num_joints)
            if p.getJointInfo(self.robotId, joint)[2] in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}
        ]

        # Extract joint limits from urdf
        self.joint_limits = [p.getJointInfo(self.robotId, i)[8:10] for i in self.controllable_joint_idx]

    def prune_arc(self, prune_point, radius, allowance_angle, num_arc_points, y_ori_default=0.0, z_ori_default=0):
        # Define theta as a descrete array
        theta = np.linspace(3 * np.pi/2 - allowance_angle, 3 * np.pi/2 + allowance_angle, num_arc_points)

        # Set up arc length coordinate
        x = np.full_like(theta, prune_point[0])  # x-coordinate remains constant
        z = - radius * np.cos(theta) + prune_point[2] # multiply by a negative to mirror on other side of axis
        y = radius * np.sin(theta) + prune_point[1] 

        # Calculate orientation angles
        arc_angles = np.arctan2(prune_point[1] - y, prune_point[2] - z)

        arc_coords = np.vstack((x, y, z))

        goal_coords = np.zeros((num_arc_points, 3)) # 3 for x, y, z
        goal_orientations = np.zeros((num_arc_points, 4)) # 4 for quaternion
        for i in range(num_arc_points):
            goal_coords[i] = [arc_coords[0][i], arc_coords[1][i], arc_coords[2][i]]
            goal_orientations[i] = p.getQuaternionFromEuler([-arc_angles[i], y_ori_default, z_ori_default])

        return goal_coords, goal_orientations
        
    def set_joint_positions(self, joint_positions):
        for i, joint_idx in enumerate(self.controllable_joint_idx):
            p.setJointMotorControl2(self.robotId, joint_idx, p.POSITION_CONTROL, joint_positions[i])
    
    def get_joint_positions(self):
        return [p.getJointState(self.robotId, i)[0] for i in self.controllable_joint_idx]
    
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
    
    def simple_controller(self, target_joint_positions, position_tol=0.1, planar=True):
        # Iterate over the joints and set their positions
        for i, joint_idx in enumerate(self.controllable_joint_idx):
            p.setJointMotorControl2(
                bodyIndex=self.robotId,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joint_positions[i]
            )

        # Wait until the manipulator reaches the target positions
        while True:
            joint_states = [p.getJointState(self.robotId, joint_idx)[0] for i, joint_idx in enumerate(self.controllable_joint_idx)]
            if all(abs(joint_states[i] - target_joint_positions[i]) < position_tol for i in range(len(target_joint_positions))):
                break
            p.stepSimulation()
            time.sleep(0.01)

        # Plot manipulator manipulability after reaching target positions
        manipulability = self.calculate_manipulability(list(target_joint_positions), planar=planar, visualize_jacobian=True)
        print(f'Manipulability: {manipulability}')

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
            
            iteration = 0
            while True:
                final_conf = generate_sample()
                self.set_joint_positions(final_conf)
                end_effector_position, _ = self.get_link_state(self.end_effector_index)
                
                if end_effector_position[1] >= 0:  # Check if y >= 0 (in front of the xz plane)
                    break  # Accept the sample if the end-effector is in front of the xz plane
                
                iteration += 1
                if iteration > 200:
                    break
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
    planar_pruner = PlanarPruner(urdf_filename="ur5e/ur5e_cutter_cart")
    prune_point = planar_pruner.prune_point_1_pos

    # prune_points = [planar_pruner.prune_point_0_pos, planar_pruner.prune_point_1_pos, planar_pruner.prune_point_2_pos]

    simple_control = False
    save_data = False
    planar = False

    # Parameters for 2D arc manipulability search
    num_arc_points = 30

    # Parameters for 3D hemisphere manipulability search
    num_hemisphere_points = [8, 8] # [num_theta, num_phi]
    look_at_point_offset = 0.1
    hemisphere_radius = 0.1
    hemisphere_center = np.copy(prune_point)
    hemisphere_center[1] -= look_at_point_offset    

    if planar:
        goal_coords, goal_orientations = planar_pruner.prune_arc(prune_point,
                                                                radius=0.1, 
                                                                allowance_angle=np.deg2rad(30), 
                                                                num_arc_points=num_arc_points, 
                                                                y_ori_default=0, 
                                                                z_ori_default=0)
    else:
        goal_coords = sample_hemisphere_suface_pts(hemisphere_center, hemisphere_radius, num_hemisphere_points)
        print(len(goal_coords))
        goal_orientations = end_effector_orientations(prune_point, goal_coords)

    if simple_control:
        # poi = int(num_arc_points / 2)
        poi = 0
        start_end_effector_pos, start_end_effector_orientation = planar_pruner.get_link_state(planar_pruner.end_effector_index)

        print(p.getEulerFromQuaternion(start_end_effector_orientation))

        # print(p.getEulerFromQuaternion(goal_orientations[poi]))
        new_goal_pos = [0.5, 0.6, 0.85]
        # goal_point = planar_pruner.load_urdf("sphere2.urdf", new_goal_pos, radius=0.05)
        new_goal_ori = p.getQuaternionFromEuler([0.5, 0, 0])
        # target_joint_positions = np.array(p.calculateInverseKinematics(planar_pruner.robotId, planar_pruner.end_effector_index, new_goal_pos, new_goal_ori))

        target_joint_positions = np.array(p.calculateInverseKinematics(planar_pruner.robotId, planar_pruner.end_effector_index, goal_coords[poi], goal_orientations[poi]))

        length = 0.1
        for i in range(num_arc_points):
            start_point = goal_coords[i]
            # print(start_point)

            # Convert back to euler (SOMETIMES HAS ABIGUITY BETWEEN -PI TO PI)
            angle = p.getEulerFromQuaternion(goal_orientations[i])
            end_point = [start_point[0],
                        start_point[1] + length * np.sin(angle[1]),
                        start_point[2] + length * np.cos(angle[1])
                        ]
            # p.addUserDebugLine(start_point, end_point, (1, 0, 0), 2)

        planar_pruner.simple_controller(target_joint_positions, position_tol=0.1, planar=False)

    else:
        if save_data:
            data_filename = "prrr_y_arc_manip_new"

            input(f"\nSaving data to: {data_filename}.csv. Press Enter if correct")

        start_end_effector_pos, start_end_effector_orientation = planar_pruner.get_link_state(planar_pruner.end_effector_index)

        start_conf = planar_pruner.inverse_kinematics(start_end_effector_pos, start_end_effector_orientation)
        controllable_joints = planar_pruner.controllable_joint_idx
        distance_fn = get_distance_fn(planar_pruner.robotId, controllable_joints)
        extend_fn = get_extend_fn(planar_pruner.robotId, controllable_joints)
        collision_fn = get_collision_fn(planar_pruner.robotId, controllable_joints, planar_pruner.collision_objects)

        sys_manipulability = np.zeros((num_arc_points, 1))

        for point in range(num_arc_points):
            # sample_fn = planar_pruner.vector_field_sample_fn(goal_coords[point], goal_orientations[point], alpha=0.6)
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
                                                                planar=False, 
                                                                max_iter=200)
            
            print(f"Highest manipulability found: {np.round(manipulability_max, 5)}")
            sys_manipulability[point] = manipulability_max

        print("Finished!\n")

        if save_data:
            np.savetxt(f"./data/{data_filename}.csv", np.hstack((goal_coords, goal_orientations, sys_manipulability)))

        p.disconnect()


if __name__ == "__main__":
    main()
