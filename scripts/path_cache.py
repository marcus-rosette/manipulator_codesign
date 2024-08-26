import numpy as np
from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot
from sample_approach_points import SamplePoints


class PathCache:
    def __init__(self, robot_urdf_path: str, planar: bool, renders=True):
        self.pyb = PybUtils(self, renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)
        self.robot = LoadRobot(self.pyb.con, robot_urdf_path, [0, -0.5, 0], self.pyb.con.getQuaternionFromEuler([0, 0, 0]))
        self.point_sampler = SamplePoints(self.pyb.con, planar)

    def find_high_manip_ik(self, points, num_hemisphere_points, look_at_point_offset, hemisphere_radius, num_configs_in_path=100, save_data_filename=None, path_filename=None):
        num_points = len(points)

        best_iks = np.zeros((num_points, len(self.robot.controllable_joint_idx)))
        best_ee_positions = np.zeros((num_points, 3))
        best_orienations = np.zeros((num_points, 4))
        best_manipulabilities = np.zeros((num_points, 1))
        best_paths = np.zeros((num_configs_in_path, len(self.robot.controllable_joint_idx), num_points))

        for i, pt in enumerate(points):
            hemisphere_pts = self.point_sampler.sample_hemisphere_suface_pts(pt, look_at_point_offset, hemisphere_radius, num_hemisphere_points)
            hemisphere_oris = self.point_sampler.hemisphere_orientations(pt, hemisphere_pts)

            best_ik = None
            best_ee_pos = None
            best_orienation = None
            best_manipulability = 0
            best_path = None

            for target_position, target_orientation in zip(hemisphere_pts, hemisphere_oris):
                joint_angles = self.robot.inverse_kinematics(target_position, target_orientation)

                manipulability = self.robot.calculate_manipulability(joint_angles, planar=False, visualize_jacobian=False)
                
                if manipulability > best_manipulability:
                    best_ik = joint_angles
                    best_ee_pos = target_position
                    best_orienation = target_orientation
                    best_manipulability = manipulability
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

    voxel_centers = np.loadtxt('./data/voxel_centers2.csv')

    save_data_filename = './data/voxel_ik_solutions2.csv'
    path_filename = './data/voxel_paths2'
    
    path_cache.find_high_manip_ik(voxel_centers, num_hemisphere_points, look_at_point_offset, hemisphere_radius, save_data_filename=save_data_filename, path_filename=path_filename)