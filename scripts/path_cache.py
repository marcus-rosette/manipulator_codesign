import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot
from sample_approach_points import sample_hemisphere_suface_pts, hemisphere_orientations


class PathCache:
    def __init__(self, robot_urdf_path: str, robot_home_pos, ik_tol=0.15, renders=True):
        """ Generate a cache of paths to high scored manipulability configurations

        Args:
            robot_urdf_path (str): filename/path to urdf file of robot
            renders (bool, optional): visualize the robot in the PyBullet GUI. Defaults to True.
        """
        self.pyb = PybUtils(self, renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)

        self.robot_home_pos = robot_home_pos
        self.robot = LoadRobot(self.pyb.con, robot_urdf_path, [0, 0, 0], self.pyb.con.getQuaternionFromEuler([0, 0, 0]), self.robot_home_pos, collision_objects=self.object_loader.collision_objects)
        
        start_position, start_orientation = self.robot.get_link_state(self.robot.end_effector_index)
        self.start_pose = np.concatenate((start_position, start_orientation))

        self.ik_tol = ik_tol

    def find_high_manip_ik(self, points, num_hemisphere_points, look_at_point_offset, hemisphere_radius, num_configs_in_path=100, save_data=True, save_data_filename=None, path_filename=None):
        """ Find the inverse kinematic solutions that result in the highest manipulability

        Args:
            points (float list): target end-effector points
            num_hemisphere_points (int list): number of points along each dimension [num_theta, num_pi]
            look_at_point_offset (float): distance to offset the sampled hemisphere from the target point
            hemisphere_radius (float): radius of generated hemisphere
            num_configs_in_path (int, optional): number of joint configurations within path. Defaults to 100.
            save_data_filename (str, optional): file name/path for saving inverse kinematics data. Defaults to None.
            path_filename (str, optional): file name/path for saving resultant paths. Defaults to None.
        """
        num_points = len(points)

        # Initialize arrays for saving data
        best_iks = np.zeros((num_points, len(self.robot.controllable_joint_idx)))
        best_ee_positions = np.zeros((num_points, 3))
        best_orienations = np.zeros((num_points, 4))
        best_manipulabilities = np.zeros((num_points, 1))
        best_paths = np.zeros((num_configs_in_path, len(self.robot.controllable_joint_idx), num_points))

        nan_mask = None
        increment = 0.05  # 5% print increment

        for i, pt in enumerate(points):
            if i % int(increment * num_points) == 0:
                print(f"{np.round(i / num_points, 2) * 100}% Complete")

            # Sample target points
            hemisphere_pts = sample_hemisphere_suface_pts(pt, look_at_point_offset, hemisphere_radius, num_hemisphere_points)
            hemisphere_oris = hemisphere_orientations(pt, hemisphere_pts)

            best_ik = None
            best_ee_pos = None
            best_orienation = None
            best_manipulability = 0
            best_path = None

            # Get IK solution for each target point on hemisphere and save the one with the highest manipulability 
            for target_position, target_orientation in zip(hemisphere_pts, hemisphere_oris):
                # target_orientation = self.robot.limit_quaternion_y_rot(self.start_pose[3:], target_orientation)

                # joint_angles = self.robot.inverse_kinematics(target_position, adjusted_target_quat)
                joint_angles = self.robot.inverse_kinematics(target_position, target_orientation)

                self.robot.reset_joint_positions(joint_angles)
                ee_pos, ee_ori = self.robot.get_link_state(self.robot.end_effector_index)

                # If the target joint anlges result in a collisions, skip the iteration
                ground_collision = self.robot.check_collision_aabb(self.robot.robotId, self.object_loader.planeId)
                # vtrellis_tree_collision = self.robot.check_collision_aabb(self.robot.robotId, self.object_loader.vtrellis_treeId)
                if ground_collision: # or vtrellis_tree_collision:
                    continue

                # If the distance between the desired point and found ik solution ee-point is greater than the tol, then skip the iteration
                distance = np.linalg.norm(ee_pos - target_position)
                if distance > self.ik_tol:
                    continue

                manipulability = self.robot.calculate_manipulability(joint_angles, planar=False, visualize_jacobian=False)
                
                if manipulability > best_manipulability:
                    # Interpolate a path from the starting configuration to the best IK solution
                    path, collision_in_path = self.robot.interpolate_joint_positions(self.robot_home_pos, joint_angles, num_configs_in_path)
                    if not collision_in_path:
                        best_path = path
                        best_ik = joint_angles
                        best_ee_pos = target_position
                        best_orienation = target_orientation
                        best_manipulability = manipulability

            best_iks[i, :] = best_ik
            best_ee_positions[i, :] = best_ee_pos
            best_orienations[i, :] = best_orienation
            best_manipulabilities[i, :] = best_manipulability
            best_paths[:, :, i] = best_path
        
        # Combine all end-effector solutions
        end_effector_solu = np.hstack((best_iks, best_ee_positions, best_orienations, best_manipulabilities))

        # Remove rows with any NaN values
        nan_mask = ~np.isnan(end_effector_solu).any(axis=1) 
        end_effector_solu_cleaned = end_effector_solu[nan_mask]
        best_paths_cleaned = best_paths[:, :, nan_mask]

        if save_data:
            # Save end effector solutions
            np.savetxt(save_data_filename, end_effector_solu_cleaned, delimiter=",", header="j1,j2,j3,j4,j5,j6,end_effector_x,end_effector_y,end_effector_z,quat_x,quat_y,quat_z,quat_w,manipulability", comments='')

            # Save associated paths as a .npy file
            np.save(path_filename, best_paths_cleaned)

            np.save('/home/marcus/apple_harvest_ws/src/apple-harvest/harvest_control/resource/reachable_paths.npy', best_paths_cleaned)

        return nan_mask

            
if __name__ == "__main__":
    render = False
    robot_home_pos = [np.pi/2, -3*np.pi/4, np.pi/2, -3*np.pi/4, -np.pi/2, 0]
    # robot_home_pos = [0, -np.pi/2, 0, -np.pi/2, 0, 0]
    path_cache = PathCache(robot_urdf_path="./urdf/ur5e/ur5e.urdf", renders=render, robot_home_pos=robot_home_pos)

    num_hemisphere_points = [8, 8] # [num_theta, num_phi]
    look_at_point_offset = 0.0
    hemisphere_radius = 0.15

    voxel_data = np.loadtxt('./data/voxel_data_parallelepiped.csv')
    voxel_centers = voxel_data[:, :3]
    # voxel_indices = voxel_data[:, 3:]

    # Translate voxels in front of robot
    y_trans = 0.65
    voxel_centers_shifted = np.copy(voxel_centers)
    voxel_centers_shifted[:, 1] += y_trans
    # voxel_centers = voxel_centers_shifted

    num_configs_in_path = 100
    save_data_filename = './data/voxel_ik_data.csv'
    path_filename = './data/reachable_paths'
    
    nan_mask = path_cache.find_high_manip_ik(voxel_centers_shifted, 
                                             num_hemisphere_points, 
                                             look_at_point_offset, 
                                             hemisphere_radius, 
                                             num_configs_in_path=num_configs_in_path, 
                                             save_data_filename=save_data_filename, 
                                             path_filename=path_filename)

    # Filtered voxel_data
    voxel_data_filtered = voxel_centers_shifted[nan_mask]
    print(voxel_data_filtered.shape)
    filename = './data/reachable_voxels_centers.csv'
    np.savetxt(filename, voxel_data_filtered)

    filename2 = '/home/marcus/apple_harvest_ws/src/apple-harvest/harvest_control/resource/reachable_voxel_centers.csv'
    np.savetxt(filename2, voxel_data_filtered)
    