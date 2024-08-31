import pybullet as p
import numpy as np
import time
from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot
from sample_approach_points import prune_arc, sample_hemisphere_suface_pts, hemisphere_orientations
from pybullet_planning import (rrt_connect, get_distance_fn, get_sample_fn, get_extend_fn, get_collision_fn)


class PrunerEnv:
    def __init__(self, robot_urdf_path: str, planar: bool, renders=True):
        self.pyb = PybUtils(self, renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)
        self.robot = LoadRobot(self.pyb.con, robot_urdf_path, [self.object_loader.start_x, 0, 0], self.pyb.con.getQuaternionFromEuler([0, 0, 0]))
        
    def simple_controller(self, target_joint_positions, position_tol=0.1, planar=True):
        # Iterate over the joints and set their positions
        self.robot.set_joint_positions(target_joint_positions)

        # Wait until the manipulator reaches the target positions
        while True:
            joint_states = self.robot.get_joint_positions()
            if all(abs(joint_states[i] - target_joint_positions[i]) < position_tol for i in range(len(target_joint_positions))):
                break
            p.stepSimulation()
            time.sleep(0.01)

        # Plot manipulator manipulability after reaching target positions
        manipulability = self.robot.calculate_manipulability(list(target_joint_positions), planar=planar, visualize_jacobian=True)
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
            self.robot.set_joint_positions(random_conf)
            end_effector_position, _ = self.robot.get_link_state(self.robot.end_effector_index)
            
            vector_to_goal = np.array(goal_position) - end_effector_position
            guided_position = end_effector_position + vector_to_goal
            # guided_conf = np.array(self.robot.inverse_kinematics(guided_position, goal_orientation))
            guided_conf = np.array(self.robot.inverse_kinematics(guided_position))
            final_conf = (1 - alpha) * random_conf + alpha * guided_conf
            
            return final_conf
        return sample
    
    def new_sample_fn(self, goal_position, goal_orientation, alpha=0.8):
        def sample():
            def generate_sample():
                # Sample a random configuration within joint limits
                random_conf = np.random.uniform([limit[0] for limit in self.robot.joint_limits], 
                                                [limit[1] for limit in self.robot.joint_limits])
                self.robot.set_joint_positions(random_conf)
                
                # Get the current end-effector position
                end_effector_position, _ = self.robot.get_link_state(self.robot.end_effector_index)
                
                # Calculate the vector towards the goal position
                vector_to_goal = np.array(goal_position) - end_effector_position
                guided_position = end_effector_position + vector_to_goal
                
                # Perform inverse kinematics to find the configuration for the guided position
                guided_conf = np.array(self.robot.inverse_kinematics(guided_position))
                
                # Blend the random configuration and guided configuration using alpha
                final_conf = (1 - alpha) * random_conf + alpha * guided_conf
                
                return final_conf
            
            iteration = 0
            while True:
                final_conf = generate_sample()
                self.robot.set_joint_positions(final_conf)
                end_effector_position, _ = self.robot.get_link_state(self.robot.end_effector_index)
                
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
                self.robot.set_joint_positions(final_joint_positions)
                final_end_effector_pos, final_end_effector_orientation = self.robot.get_link_state(self.robot.end_effector_index)

                within_tol = self.robot.check_pose_within_tolerance(final_end_effector_pos, final_end_effector_orientation, goal_coord, goal_ori, pos_tol, ori_tol)
                if within_tol:

                    # Calculate and print manipulability
                    manipulability = self.robot.calculate_manipulability(final_joint_positions, planar=planar)
                    if manipulability > manipulability_max:
                        manipulability_max = manipulability
                        path_best = path
                else:
                    path = None

        return manipulability_max, path_best
    

def main():
    simple_control = False
    save_data = True
    planar = True
    rrt_control = False

    pruner_env = PrunerEnv(robot_urdf_path="./urdf/ur5e/ur5e_cart.urdf", planar=planar, renders=False)
    prune_point = pruner_env.object_loader.prune_point_1_pos

    # Parameters for 2D arc manipulability search
    num_arc_points = 60

    # Parameters for 3D hemisphere manipulability search
    num_hemisphere_points = [2, 2] # [num_theta, num_phi]
    look_at_point_offset = 0.1
    hemisphere_radius = 0.1  

    if planar:
        goal_coords, goal_orientations = prune_arc(prune_point,
                                                    radius=0.1, 
                                                    allowance_angle=np.deg2rad(30), 
                                                    num_arc_points=num_arc_points, 
                                                    y_ori_default=0, 
                                                    z_ori_default=0)
        num_points = num_arc_points
        sys_manipulability = np.zeros((num_points, 1))
    else:
        goal_coords = sample_hemisphere_suface_pts(prune_point, look_at_point_offset, hemisphere_radius, num_hemisphere_points)
        goal_orientations = hemisphere_orientations(prune_point, goal_coords)
        num_points = len(goal_coords)
        sys_manipulability = np.zeros((num_points, 1))

    if simple_control:
        poi = 0
        start_end_effector_pos, start_end_effector_orientation = pruner_env.robot.get_link_state(pruner_env.robot.end_effector_index)

        new_goal_pos = [0.5, 0.6, 0.85]
        new_goal_ori = pruner_env.pyb.con.getQuaternionFromEuler([0.5, 0, 0])

        target_joint_positions = np.array(pruner_env.robot.inverse_kinematics(goal_coords[poi], goal_orientations[poi]))

        length = 0.1
        for i in range(num_points):
            start_point = goal_coords[i]

            # Convert back to euler (SOMETIMES HAS ABIGUITY BETWEEN -PI TO PI)
            angle = pruner_env.pyb.con.getEulerFromQuaternion(goal_orientations[i])
            end_point = [start_point[0],
                        start_point[1] + length * np.sin(angle[1]),
                        start_point[2] + length * np.cos(angle[1])
                        ]
            # p.addUserDebugLine(start_point, end_point, (1, 0, 0), 2)

        pruner_env.simple_controller(target_joint_positions, position_tol=0.1, planar=False)

    else:
        if save_data:
            data_filename = "ur5_planar_arc_manip"

            input(f"\nSaving data to: {data_filename}.csv. Press Enter if correct")

        start_end_effector_pos, start_end_effector_orientation = pruner_env.robot.get_link_state(pruner_env.robot.end_effector_index)

        start_conf = pruner_env.robot.inverse_kinematics(start_end_effector_pos, start_end_effector_orientation)
        controllable_joints = pruner_env.robot.controllable_joint_idx
        distance_fn = get_distance_fn(pruner_env.robot.robotId, controllable_joints)
        extend_fn = get_extend_fn(pruner_env.robot.robotId, controllable_joints)
        collision_fn = get_collision_fn(pruner_env.robot.robotId, controllable_joints, pruner_env.object_loader.collision_objects)

        for point in range(num_points):
            # sample_fn = pruner_env.vector_field_sample_fn(goal_coords[point], goal_orientations[point], alpha=0.6)
            sample_fn = pruner_env.new_sample_fn(goal_coords[point], goal_orientations[point], alpha=0.6)
            # sample_fn = get_sample_fn(pruner_env.robotId, controllable_joints)

            goal_conf = pruner_env.robot.inverse_kinematics(goal_coords[point], goal_orientations[point])

            if rrt_control:
                manipulability_max, path_best = pruner_env.rrtc_loop(goal_coords[point], goal_orientations[point], 
                                                                    start_conf, goal_conf, 
                                                                    sample_fn, 
                                                                    distance_fn, 
                                                                    extend_fn, 
                                                                    collision_fn, 
                                                                    pos_tol=0.1,
                                                                    ori_tol=0.5,
                                                                    planar=False, 
                                                                    max_iter=500)

            else:
                manipulability_max = pruner_env.robot.calculate_manipulability(goal_conf, planar=False)
            
            print(f"Highest manipulability found: {np.round(manipulability_max, 5)}")
            sys_manipulability[point] = manipulability_max

        print("Finished!\n")

        if save_data:
            np.savetxt(f"./data/{data_filename}.csv", np.hstack((goal_coords, goal_orientations, sys_manipulability)))

        pruner_env.pyb.disconnect()


if __name__ == "__main__":
    main()
