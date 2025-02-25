import numpy as np
import time
from scipy.spatial.transform import Rotation
from .pyb_utils import PybUtils
from .load_objects import LoadObjects
from .load_robot import LoadRobot


class ViewRobot:
    def __init__(self, robot_urdf_path: str, robot_home_pos, ik_tol=0.01, renders=True, ee_link_name='ee_link'):
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
                               collision_objects=self.object_loader.collision_objects,
                               ee_link_name=ee_link_name)

        self.ik_tol = ik_tol

    def cartesian_path_test(self):
        # Euler angles to point the end-effector in the y-direction
        orientation = list(self.pyb.con.getQuaternionFromEuler([-np.pi/2, 0, 0]))

        # Define start and end positions
        start_pos = [0.5, 0.75, 0.5]
        end_pos = [-0.5, 0.75, 0.5]

        # Create start and end poses as (position, orientation)
        start_pose = (start_pos, orientation)
        end_pose = (end_pos, orientation)

        print("Start Pose:", start_pose)
        print("End Pose:", end_pose)

        # Interpolate a list of waypoint poses between start_pose and end_pose
        num_points = 10
        waypoints = []
        for t in np.linspace(0, 1, num_points):
            interp_pos = (np.array(start_pos) * (1 - t) + np.array(end_pos) * t).tolist()

            # Orientation remains constant
            waypoints.append((interp_pos, orientation))

        # Call the plan_cartesian_motion_path function with the list of waypoint poses
        joint_path = self.robot.plan_cartesian_motion_path(waypoints, max_iterations=10000)

        if joint_path is None:
            print("Cartesian path planning failed.")
            return

        # Execute the planned joint path
        for joint_config in joint_path:
            self.robot.set_joint_positions(joint_config)
            for _ in range(240):
                self.pyb.con.stepSimulation()
                time.sleep(1. / 240.)

    def main(self):
        target_positions = [
            [1.0, 1.0, 1.0],
            [0.5, 1.5, 1.2],
            [-0.5, 1.0, 0.8],
            [0.0, 1.0, 1.0],
            [-1.0, 1.5, 1.2],
            [-0.5, 0.5, 0.1],
            [0.5, 0.5, 1.0],
            [1.2, -0.5, 0.9],
            [-1.2, 0.8, 1.1],
            [0.3, -1.0, 1.3],
            [-0.7, -1.2, 0.7],
            [0.8, 1.3, 1.4],
            [-1.1, -0.8, 0.6],
            [0.6, -0.6, 1.5],
            [-0.3, 0.7, 1.2]
        ]
        target_orientations = [
            Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [-90, 90, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
            Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat()
        ]
        target_poses = list(zip(target_positions, target_orientations))

        for pose in target_positions:
            input("Press Enter to continue...")
            self.object_loader.load_urdf("sphere2.urdf", 
                                        start_pos=pose, 
                                        start_orientation=[0, 0, 0], 
                                        fix_base=True, 
                                        radius=0.05)
        
            joint_config = self.robot.inverse_kinematics(pose, pos_tol=self.ik_tol)
            self.robot.set_joint_positions(joint_config)

            # Step simulation and render
            for _ in range(240):  # Adjust number of simulation steps as needed
                self.pyb.con.stepSimulation()
                time.sleep(1./240.)  # Sleep to match real-time
        
        while True:
            self.pyb.con.stepSimulation()

            
if __name__ == "__main__":
    robot_urdf_path = "/home/marcus/IMML/manipulator_codesign/manipulator_codesign/urdf/robots/best_chain_pyb_GA_15poses_full_collision.urdf"
    ee_link_name = 'end_effector' # for best_chain
    # robot_urdf_path = '/home/marcus/IMML/manipulator_codesign/manipulator_codesign/urdf/robots/ur5e/ur5e.urdf'
    # ee_link_name = 'gripper_link' # for ur5e
    render = True
    robot_home_pos = None #[0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    view_robot = ViewRobot(robot_urdf_path=robot_urdf_path, 
                        renders=render, 
                        robot_home_pos=robot_home_pos,
                        ik_tol=0.1,
                        ee_link_name=ee_link_name)
    
    # view_robot.cartesian_path_test()
    view_robot.main()
