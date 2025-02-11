import numpy as np
import sys
import os

# Get the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot


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
        self.robot = LoadRobot(self.pyb.con, 
                               robot_urdf_path, 
                               [0, 0, 0], 
                               self.pyb.con.getQuaternionFromEuler([0, 0, 0]), 
                               self.robot_home_pos, 
                               collision_objects=self.object_loader.collision_objects)

        self.ik_tol = ik_tol

    def main(self):
        target_positions = [
            [1.0, 2.0, 1.0],
            [0.5, 1.5, 1.2],
            [-0.5, 1.0, 0.8],
            [0.0, 1.0, 1.0],
            [-1.0, 1.5, 1.2],
            [-0.5, 0.5, 0.1],
            [0.5, 0.5, 0.1],
        ]
        # target_point = target_positions[0]
        target_point = [1.5, 0.2, 0.3]
        self.object_loader.load_urdf("sphere2.urdf", 
                                     start_pos=target_point, 
                                     start_orientation=[0, 0, 0], 
                                     fix_base=True, 
                                     radius=0.05)
        
        joint_config = self.robot.inverse_kinematics(target_point, pos_tol=self.ik_tol)
        print(joint_config)
        self.robot.set_joint_positions(joint_config)

        while True:
            self.pyb.con.stepSimulation()

            
if __name__ == "__main__":
    render = True
    robot_home_pos = [0, 0, 0, 0, 0]
    # path_cache = PathCache(robot_urdf_path="/home/marcus/IMML/manipulator_codesign/manipulator_codesign/gen_urdf_files/best_chain_10.urdf", 
    #                        renders=render, 
    #                        robot_home_pos=robot_home_pos)
    path_cache = PathCache(robot_urdf_path="/home/marcus/IMML/manipulator_codesign/manipulator_codesign/urdf/test_robot.urdf", 
                        renders=render, 
                        robot_home_pos=robot_home_pos)
    
    path_cache.main()
