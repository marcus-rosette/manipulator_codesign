import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from pybullet_planning import (rrt_connect, get_distance_fn, get_sample_fn, get_extend_fn, get_collision_fn)


class PlanarPruner:
    def __init__(self):
        # Connect to the PyBullet physics engine with GUI
        self.physicsClient = p.connect(p.GUI)

        # Set the path for additional URDF and other data files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set the gravity
        p.setGravity(0, 0, -10)

        # Set the camera parameters
        camera_distance = 2                      # Distance from the target
        camera_yaw = 90                          # Yaw angle in degrees
        camera_pitch = -10                       # Pitch angle in degrees
        camera_target_position = [0, 0.75, 0.75] # Target position

        # Reset the debug visualizer camera
        p.resetDebugVisualizerCamera(
            camera_distance,
            camera_yaw,
            camera_pitch,
            camera_target_position
        )

        # Load a plane URDF 
        self.planeId = p.loadURDF("plane.urdf")

        # Index of gripper (from urdf)
        self.gripper_idx = 6

        self.start_sim()

    def load_urdf(self, urdf_name, start_pos=[0, 0, 0], start_orientation=[0, 0, 0], color=None, fix_base=True, radius=None):
        orientation = p.getQuaternionFromEuler(start_orientation)

        if radius is None:
            objectId = p.loadURDF(urdf_name, start_pos, orientation, useFixedBase=fix_base)
        else:
            # Plot points as green
            objectId = p.loadURDF(urdf_name, start_pos, globalScaling=radius, useFixedBase=fix_base)
            p.changeVisualShape(objectId, -1, rgbaColor=[0, 1, 0, 1]) 

        return objectId

    def start_sim(self):
        # Define the starting position of the branches
        start_x = 0.5
        start_y = 1

        # Define the starting position of the pruning points
        self.prune_point_0_pos = [start_x, start_y, 1.55] 
        self.prune_point_1_pos = [start_x, start_y - 0.05, 1.05] 
        self.prune_point_2_pos = [start_x, start_y + 0.05, 0.55] 
        self.radius = 0.05 

        # Load the branches
        self.leader_branchId = self.load_urdf("./urdf/leader_branch.urdf", [0, start_y, 1.6/2])
        self.top_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1.5], [0, np.pi / 2, 0])
        self.mid_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1], [0, np.pi / 2, 0])
        self.bottom_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 0.5], [0, np.pi / 2, 0])
        
        # Get list of collidable objects
        self.collision_objects = [self.leader_branchId, self.top_branchId, self.mid_branchId, self.bottom_branchId, self.planeId]

        # Load the pruning points
        self.prune_point_0 = self.load_urdf("sphere2.urdf", self.prune_point_0_pos, radius=self.radius)
        self.prune_point_1 = self.load_urdf("sphere2.urdf", self.prune_point_1_pos, radius=self.radius)
        self.prune_point_2 = self.load_urdf("sphere2.urdf", self.prune_point_2_pos, radius=self.radius)

        # Get manipulator
        self.robotId = p.loadURDF("./urdf/rrr_manipulator.urdf", [start_x, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robotId)
        self.num_static_joints = 4 # static joints in the end-effector
        self.num_controllable_joints = self.num_joints - self.num_static_joints
        self.joint_limits = [p.getJointInfo(self.robotId, i)[8:10] for i in range(self.num_controllable_joints)]

    def set_joint_positions(self, joint_positions):
        for i in range(self.num_controllable_joints):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, joint_positions[i])
    
    def get_joint_positions(self):
        return [p.getJointState(self.robotId, i)[0] for i in range(self.num_controllable_joints)]
    
    def is_collision(self):
        return len(p.getContactPoints(self.robotId)) > 0
    
    def inverse_kinematics(self, position, orientation):
        joint_positions = p.calculateInverseKinematics(self.robotId, self.gripper_idx, position, orientation)
        return joint_positions
    
    def check_pose_within_tolerance(self, final_position, final_orientation, target_position, target_orientation, pos_tolerance, ori_tolerance):
        pos_diff = np.linalg.norm(np.array(final_position) - np.array(target_position))
        ori_diff = np.linalg.norm(R.from_quat(final_orientation).as_rotvec() - R.from_quat(target_orientation).as_rotvec())
        return pos_diff <= pos_tolerance and ori_diff <= ori_tolerance

def main():
    # Initialize the planar pruner
    planar_pruner = PlanarPruner()

    # Define the goal position and orientation
    goal_position = planar_pruner.prune_point_1_pos
    goal_orientation = p.getQuaternionFromEuler([-1.57, 0, 0])

    # Define tolerances
    pos_tolerance = 0.025
    ori_tolerance = 0.5  # In radians

    # Define the start position for the end-effector (assuming straight up configuration)
    start_position = [0.0, 0.0, 0.0]
    start_orientation = [0, 0, 0]

    # Get the start joint configuration using inverse kinematics
    start_conf = planar_pruner.inverse_kinematics(start_position, p.getQuaternionFromEuler(start_orientation))

    controllable_joints = list(range(planar_pruner.num_controllable_joints))
    distance_fn = get_distance_fn(planar_pruner.robotId, controllable_joints)
    sample_fn = get_sample_fn(planar_pruner.robotId, controllable_joints)
    extend_fn = get_extend_fn(planar_pruner.robotId, controllable_joints)
    collision_fn = get_collision_fn(planar_pruner.robotId, controllable_joints, planar_pruner.collision_objects)

    max_iterations = 2000
    path = None

    goal_conf = planar_pruner.inverse_kinematics(goal_position, goal_orientation)
    for i in range(max_iterations):
        path = rrt_connect(
            start_conf, goal_conf,
            extend_fn=extend_fn,
            collision_fn=collision_fn,
            distance_fn=distance_fn,
            sample_fn=sample_fn,
            max_iterations=5000
        )

        if path is not None:
            final_joint_positions = path[-1]
            planar_pruner.set_joint_positions(final_joint_positions)
            final_position, final_orientation = p.getLinkState(planar_pruner.robotId, planar_pruner.gripper_idx)[:2] # getting position and orientation

            if planar_pruner.check_pose_within_tolerance(final_position, final_orientation, goal_position, goal_orientation, pos_tolerance, ori_tolerance):
                iteration = i
                break
            else:
                path = None  # Reset path if not within tolerance

    if path is None:
        print("No path found within the specified tolerances!")
    else:
        print(f"Path found within tolerances on iteration {iteration}! Executing path...")
        for config in path:
            planar_pruner.set_joint_positions(config)
            p.stepSimulation()
            time.sleep(0.1)
        print("Path execution complete. Robot should stay at the goal position.")
        
        # Maintain the final position
        final_position = path[-1]
        while True:
            planar_pruner.set_joint_positions(final_position)
            p.stepSimulation()
            time.sleep(0.1)

    p.disconnect()

if __name__ == "__main__":
    main()
