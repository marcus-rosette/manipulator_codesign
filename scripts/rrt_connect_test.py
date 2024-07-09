import pybullet as p
import pybullet_data
import numpy as np
import time
from pybullet_planning import (
    connect, disconnect, wait_for_user, load_pybullet, set_joint_positions,
    joints_from_names, rrt_connect, plan_joint_motion,
    get_distance_fn, get_sample_fn, get_extend_fn, get_collision_fn
)

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
        self.prune_point_1_pos = [start_x, start_y - 0.05, 1.25] 
        self.prune_point_2_pos = [start_x, start_y + 0.05, 0.55] 
        self.radius = 0.05 

        # Load the branches
        self.leader_branchId = self.load_urdf("./urdf/leader_branch.urdf", [0, start_y, 1.6/2])
        self.top_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1.5], [0, np.pi / 2, 0])
        self.mid_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1], [0, np.pi / 2, 0])
        self.bottom_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 0.5], [0, np.pi / 2, 0])

        # Load the pruning points
        self.prune_point_1 = self.load_urdf("sphere2.urdf", self.prune_point_1_pos, radius=self.radius)
        self.prune_point_2 = self.load_urdf("sphere2.urdf", self.prune_point_2_pos, radius=self.radius)

        # Get manipulator
        self.robotId = p.loadURDF("./urdf/three_link_manipulator.urdf", [start_x, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robotId)
        self.num_controllable_joints = self.num_joints - 4  # 4 for the static joints in the end-effector
        self.joint_limits = [p.getJointInfo(self.robotId, i)[8:10] for i in range(self.num_controllable_joints)]

    def set_joint_positions(self, joint_positions):
        for i in range(self.num_controllable_joints):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, joint_positions[i])
    
    def get_joint_positions(self):
        return [p.getJointState(self.robotId, i)[0] for i in range(self.num_controllable_joints)]
    
    def is_collision(self):
        return len(p.getContactPoints(self.robotId)) > 0
    
    def inverse_kinematics(self, position):
        joint_positions = p.calculateInverseKinematics(self.robotId, 6, position)
        return joint_positions

def main():
    # Initialize the planar pruner
    planar_pruner = PlanarPruner()

    # Define the start and goal positions for the end-effector
    start_pos = [0.0, 0.0, 0.0]
    goal_pos = planar_pruner.prune_point_1_pos

    # Get the start and goal joint configurations using inverse kinematics
    start_conf = planar_pruner.inverse_kinematics(start_pos)
    goal_conf = planar_pruner.inverse_kinematics(goal_pos)

    controllable_joints = list(range(planar_pruner.num_controllable_joints))
    distance_fn = get_distance_fn(planar_pruner.robotId, controllable_joints)
    sample_fn = get_sample_fn(planar_pruner.robotId, controllable_joints)
    extend_fn = get_extend_fn(planar_pruner.robotId, controllable_joints)
    collision_fn = get_collision_fn(planar_pruner.robotId, controllable_joints, [planar_pruner.bottom_branchId, planar_pruner.mid_branchId, planar_pruner.top_branchId, planar_pruner.planeId])

    # Use the pybullet_planning RRT-Connect to plan a path in joint space
    path = rrt_connect(
        start_conf, goal_conf,
        extend_fn=extend_fn,
        collision_fn=collision_fn,
        distance_fn=distance_fn,
        sample_fn=sample_fn,
        max_iterations=5000
    )

    if path is None:
        print("No path found!")
    else:
        print("Path found! Executing path...")
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
            time.sleep(0.05)

    p.disconnect()

if __name__ == "__main__":
    main()
