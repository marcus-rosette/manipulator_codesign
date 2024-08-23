import pybullet as p
import numpy as np
from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot
from sample_approach_points import SamplePoints


class PathCache:
    def __init__(self, robot_urdf_path: str, planar: bool, renders=True):
        self.pyb = PybUtils(self, renders=False)
        self.object_loader = LoadObjects(self.pyb.con)
        self.robot = LoadRobot(self.pyb.con, robot_urdf_path, [self.object_loader.start_x, 0, 0], self.pyb.con.getQuaternionFromEuler([0, 0, 0]))
        self.point_sampler = SamplePoints(self.pyb.con, planar)

    def find_high_manip_ik(self, points, num_hemisphere_points, look_at_point_offset, hemisphere_radius, save_data_filename=None):
        num_points = len(points)

        best_iks = np.zeros((num_points, len(path_cache.robot.controllable_joint_idx)))
        best_ee_positions = np.zeros((num_points, 3))
        best_orienations = np.zeros((num_points, 4))
        best_manipulabilities = np.zeros((num_points, 1))

        for i, pt in enumerate(reachable_points):
            hemisphere_pts = path_cache.point_sampler.sample_hemisphere_suface_pts(pt, look_at_point_offset, hemisphere_radius, num_hemisphere_points)
            hemisphere_oris = path_cache.point_sampler.hemisphere_orientations(pt, hemisphere_pts)

            best_ik = None
            best_ee_pos = None
            best_orienation = None
            best_manipulability = 0

            for target_position, target_orientation in zip(hemisphere_pts, hemisphere_oris):
                joint_angles = path_cache.robot.inverse_kinematics(target_position, target_orientation)

                manipulability = path_cache.robot.calculate_manipulability(joint_angles, planar=False, visualize_jacobian=False)
                
                if manipulability > best_manipulability:
                    best_ik = joint_angles
                    best_ee_pos = target_position
                    best_orienation = target_orientation
                    best_manipulability = manipulability

            best_iks[i, :] = best_ik
            best_ee_positions[i, :] = best_ee_pos
            best_orienations[i, :] = best_orienation
            best_manipulabilities[i, :] = best_manipulability

        if save_data_filename is not None:
            all_data = np.hstack((best_iks, best_ee_positions, best_orienations, best_manipulabilities))
            
            # Save the array to a CSV file
            np.savetxt(save_data_filename, all_data, delimiter=",", header="ik,ee_pos,orientation,manipulability", comments='')

            
if __name__ == "__main__":
    path_cache = PathCache(robot_urdf_path="./urdf/ur5e/ur5e_cart.urdf", planar=False)

    num_hemisphere_points = [8, 8] # [num_theta, num_phi]
    look_at_point_offset = 0.1
    hemisphere_radius = 0.1

    num_reachable_points = 5
    reachable_points = np.random.uniform(low=-0.5, high=1.5, size=(num_reachable_points, 3))

    data_filename = './data/test.csv'

    path_cache.find_high_manip_ik(reachable_points, num_hemisphere_points, look_at_point_offset, hemisphere_radius, save_data_filename=data_filename)


