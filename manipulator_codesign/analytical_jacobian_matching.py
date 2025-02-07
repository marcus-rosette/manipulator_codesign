import numpy as np
from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot
from sample_approach_points import sample_hemisphere_suface_pts, hemisphere_orientations


class PathCache:
    def __init__(self, robot_urdf_path: str, robot_home_pos, ik_tol=0.05, renders=True):
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

    def get_jacobians(self, target_pts):
        # joint_configs = np.zeros((len(target_pts), self.robot.num_joints - 1))
        # jacobians = np.zeros((len(target_pts), 6, self.robot.num_joints - 1))

        joint_configs = np.zeros((len(target_pts), 6))
        jacobians = np.zeros((len(target_pts), 6, 6))

        for i, pt in enumerate(target_pts):
            joint_config = self.robot.inverse_kinematics(pt)
            jacobian = self.robot.get_jacobian(joint_config)

            print(joint_config)

            joint_configs[i, :] = joint_config
            jacobians[i, :, :] = jacobian

        np.save('/home/marcus/IMML/playground_py/data/jacobians_3d.npy', jacobians)
        np.save('/home/marcus/IMML/playground_py/data/joint_configs_3d.npy', joint_configs)

            
if __name__ == "__main__":
    render = True
    # robot_home_pos = [0, 0, 0]
    robot_home_pos = [np.pi/2, -np.pi/2, 2*np.pi/3, 5*np.pi/6, -np.pi/2, 0]
    path_cache = PathCache(robot_urdf_path="./urdf/ur5e/ur5e.urdf", renders=render, robot_home_pos=robot_home_pos)
    
    target_points = np.random.rand(100, 3)
    target_points[:, 0] = 0

    nan_mask = path_cache.get_jacobians(target_points)
