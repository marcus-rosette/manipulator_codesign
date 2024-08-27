import numpy as np
from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot
from sample_approach_points import sample_hemisphere_suface_pts, hemisphere_orientations


class PathCache:
    def __init__(self, robot_urdf_path: str, renders=True):
        """ Generate a cache of paths to high scored manipulability configurations

        Args:
            robot_urdf_path (str): filename/path to urdf file of robot
            renders (bool, optional): visualize the robot in the PyBullet GUI. Defaults to True.
        """
        self.pyb = PybUtils(self, renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)
        self.robot = LoadRobot(self.pyb.con, robot_urdf_path, [0, -0.5, 0], self.pyb.con.getQuaternionFromEuler([0, 0, 0]))

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

        # Initialize arrays for saving data
        best_iks = np.zeros((num_points, len(self.robot.controllable_joint_idx)))
        best_ee_positions = np.zeros((num_points, 3))
        best_orienations = np.zeros((num_points, 4))
        best_manipulabilities = np.zeros((num_points, 1))
        best_paths = np.zeros((num_configs_in_path, len(self.robot.controllable_joint_idx), num_points))

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

                manipulability = self.robot.calculate_manipulability(joint_angles, planar=False, visualize_jacobian=False)
                
                if manipulability > best_manipulability:
                    best_ik = joint_angles
                    best_ee_pos = target_position
                    best_orienation = target_orientation
                    best_manipulability = manipulability

                    # Interpolate a path from the starting configuration to the best IK solution
                    best_path = self.robot.linear_interp_path(joint_angles, steps=num_configs_in_path)

            best_iks[i, :] = best_ik
            best_ee_positions[i, :] = best_ee_pos
            best_orienations[i, :] = best_orienation
            best_manipulabilities[i, :] = best_manipulability
            best_paths[:, :, i] = best_path

        if save_data_filename is not None:
            all_data = np.hstack((best_iks, best_ee_positions, best_orienations, best_manipulabilities))
            
            # Save the array to a CSV file
            np.savetxt(save_data_filename, all_data, delimiter=",", header="ik,ee_pos,orientation,manipulability", comments='')
        
        if path_filename is not None:
            np.save(path_filename, best_paths)

            
if __name__ == "__main__":
    render = False
    path_cache = PathCache(robot_urdf_path="./urdf/ur5e/ur5e_cart.urdf", planar=False, renders=render)

    num_hemisphere_points = [8, 8] # [num_theta, num_phi]
    look_at_point_offset = 0.1
    hemisphere_radius = 0.1

    voxel_data = np.loadtxt('./data/voxel_data_parallelepiped.csv')
    voxel_centers = voxel_data[:, :3]
    voxel_indices = voxel_data[:, 3:]

    save_data_filename = './data/voxel_ik_solutions_parallelepiped.csv'
    path_filename = './data/voxel_paths_parallelepiped'
    
    path_cache.find_high_manip_ik(voxel_centers, num_hemisphere_points, look_at_point_offset, hemisphere_radius, save_data_filename=save_data_filename, path_filename=path_filename)