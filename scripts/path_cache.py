import numpy as np
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
        self.robot = LoadRobot(self.pyb.con, robot_urdf_path, [0, 0, 0], self.pyb.con.getQuaternionFromEuler([0, 0, 0]), self.robot_home_pos)

        self.ik_tol = ik_tol

    def find_high_manip_ik(self, points, num_hemisphere_points, look_at_point_offset, hemisphere_radius, num_configs_in_path=100, save_data_filename=None, path_filename=None):
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

        # for pt in points:
        #     self.object_loader.load_urdf('sphere2.urdf', start_pos=pt, radius=0.02)

        # Initialize arrays for saving data
        best_iks = np.zeros((num_points, len(self.robot.controllable_joint_idx)))
        best_ee_positions = np.zeros((num_points, 3))
        best_orienations = np.zeros((num_points, 4))
        best_manipulabilities = np.zeros((num_points, 1))
        best_paths = np.zeros((num_configs_in_path, len(self.robot.controllable_joint_idx), num_points))

        nan_mask = None

        for i, pt in enumerate(points):
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
                joint_angles = self.robot.inverse_kinematics(target_position, target_orientation)

                self.robot.reset_joint_positions(joint_angles)
                link_pos, _ = self.robot.get_link_state(self.robot.end_effector_index)

                distance = np.linalg.norm(link_pos - target_position)

                # If the distance between the desired point and found ik solution ee-point, then skip the iteration
                if distance > self.ik_tol:
                    continue

                manipulability = self.robot.calculate_manipulability(joint_angles, planar=False, visualize_jacobian=False)
                
                if manipulability > best_manipulability:
                    best_ik = joint_angles
                    best_ee_pos = target_position
                    best_orienation = target_orientation
                    best_manipulability = manipulability

                    # Interpolate a path from the starting configuration to the best IK solution
                    best_path = self.robot.linear_interp_end_effector_path(self.robot_home_pos, joint_angles, steps=num_configs_in_path)

            best_iks[i, :] = best_ik
            best_ee_positions[i, :] = best_ee_pos
            best_orienations[i, :] = best_orienation
            best_manipulabilities[i, :] = best_manipulability
            best_paths[:, :, i] = best_path

        if save_data_filename is not None:
            all_data = np.hstack((best_iks, best_ee_positions, best_orienations, best_manipulabilities))

            # Remove rows with any NaN values
            nan_mask = ~np.isnan(all_data).any(axis=1)
            cleaned_data = all_data[nan_mask]

            # Save the array to a CSV file
            np.savetxt(save_data_filename, cleaned_data, delimiter=",", header="ik,ee_pos,orientation,manipulability", comments='')
        
        if path_filename is not None:
            # Identify which slices (rows along the 3rd dimension) contain NaN values
            nan_mask_3d = ~np.isnan(best_paths).any(axis=(0, 1))

            # Remove the slices that contain NaN values
            cleaned_path_data = best_paths[:, :, nan_mask_3d]

            # Save the array to a npy file
            np.save(path_filename, cleaned_path_data)

        return nan_mask

            
if __name__ == "__main__":
    render = False
    robot_home_pos = [np.pi/2, -np.pi/3, 2*np.pi/3, 2*np.pi/3, -np.pi/2, 0]
    path_cache = PathCache(robot_urdf_path="./urdf/ur5e/ur5e.urdf", renders=render, robot_home_pos=robot_home_pos)

    num_hemisphere_points = [8, 8] # [num_theta, num_phi]
    look_at_point_offset = 0.1
    hemisphere_radius = 0.1

    voxel_data = np.loadtxt('./data/voxel_data_parallelepiped.csv')
    voxel_centers = voxel_data[:, :3]
    voxel_indices = voxel_data[:, 3:]

    # Translate voxels in front of robot
    y_trans = 0.45
    voxel_centers_shifted = np.copy(voxel_centers)
    voxel_centers_shifted[:, 1] += y_trans
    voxel_centers = voxel_centers_shifted

    save_data_filename = './data/voxel_ik_solutions_parallelepiped.csv'
    path_filename = './data/voxel_paths_parallelepiped'

    # save_data_filename = None
    # path_filename = None
    
    nan_mask = path_cache.find_high_manip_ik(voxel_centers, num_hemisphere_points, look_at_point_offset, hemisphere_radius, save_data_filename=save_data_filename, path_filename=path_filename)

    # Filtered voxel_data
    voxel_data_filtered = voxel_data[nan_mask]
    filename = './data/voxel_data_parallelepiped_filtered.csv'
    np.savetxt(filename, voxel_data_filtered)