import numpy as np
import time
from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot


class ViewRobot:
    def __init__(self, robot_urdf_path: str, robot_home_pos, ik_tol=0.01, renders=True):
        """
        Initialize the ViewRobot class.

        Args:
            robot_urdf_path (str): Path to the URDF file of the robot.
            robot_home_pos (list): Home position of the robot joints.
            ik_tol (float, optional): Tolerance for inverse kinematics. Defaults to 0.01.
            renders (bool, optional): Whether to visualize the robot in the PyBullet GUI. Defaults to True.
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
        for pt in target_positions:
            input("Press Enter to continue...")
            self.object_loader.load_urdf("sphere2.urdf", 
                                        start_pos=pt, 
                                        start_orientation=[0, 0, 0], 
                                        fix_base=True, 
                                        radius=0.05)
        
            joint_config = self.robot.inverse_kinematics(pt, pos_tol=self.ik_tol)
            self.robot.set_joint_positions(joint_config)

            # Step simulation and render
            for _ in range(240):  # Adjust number of simulation steps as needed
                self.pyb.con.stepSimulation()
                time.sleep(1./240.)  # Sleep to match real-time
        
        while True:
            self.pyb.con.stepSimulation()

            
if __name__ == "__main__":
    render = True
    robot_home_pos = [0, 0, 0, 0, 0]
    view_robot = ViewRobot(robot_urdf_path="/home/marcus/IMML/manipulator_codesign/manipulator_codesign/urdf/robots/test_robot.urdf", 
                        renders=render, 
                        robot_home_pos=robot_home_pos)
    
    view_robot.main()
