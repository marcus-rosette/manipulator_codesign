import pybullet as p
import time
import pybullet_data
import numpy as np


class PlanarPruner:
    def __init__(self):
        # Connect to the PyBullet physics engine
        # p.GUI opens a graphical user interface for visualization
        # p.DIRECT runs in non-graphical mode (headless)
        self.physicsClient = p.connect(p.GUI)

        # Set the path for additional URDF and other data files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set the gravity
        p.setGravity(0, 0, -10)

        # Set the camera parameters
        camera_distance = 2                 # Distance from the target
        camera_yaw = 90                     # Yaw angle in degrees
        camera_pitch = -10                  # Pitch angle in degrees
        camera_target_position = [0, 0.75, 0.75]  # Target position

        # Reset the debug visualizer camera
        p.resetDebugVisualizerCamera(
            camera_distance,
            camera_yaw,
            camera_pitch,
            camera_target_position
        )

        # Load a plane URDF 
        self.planeId = p.loadURDF("plane.urdf")

    def load_urdf(self, urdf_name, start_pos=[0, 0, 0], start_orientation=[0, 0, 0], color=None, fix_base=True, radius=None):
        orientation = p.getQuaternionFromEuler(start_orientation)

        if radius is None:
            objectId = p.loadURDF(urdf_name, start_pos, orientation, useFixedBase=fix_base)

        else:
            # Plot points as green
            objectId = p.loadURDF(urdf_name, start_pos, globalScaling=radius, useFixedBase=fix_base)
            p.changeVisualShape(objectId, -1, rgbaColor=[0, 1, 0, 1]) 

        return objectId
    

def main():
    planar_pruner = PlanarPruner()
    sim_rate = 240 # Hz 
    num_steps = 10000 # number of steps in the simulation

    # Define the starting position of the branches
    start_x = 0.5
    start_y = 1

    # Define the starting position of the pruning points
    prune_point_0_pos = [start_x, start_y, 1.55] 
    prune_point_1_pos = [start_x, start_y - 0.05, 1.05] 
    prune_point_2_pos = [start_x, start_y + 0.05, 0.55] 
    radius = 0.05 

    # Load the manipulator
    manipulatorId = planar_pruner.load_urdf("./urdf/6r_manipulator.urdf", [start_x, 0, 0])

    # Set initial joint positions
    initial_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example initial positions for the three joints
    num_joints = p.getNumJoints(manipulatorId)
    for i in range(3):
        p.resetJointState(manipulatorId, i, initial_positions[i])

    # Load the branches
    leader_branchId = planar_pruner.load_urdf("./urdf/leader_branch.urdf", [0, start_y, 1.6/2])
    top_branchId = planar_pruner.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1.5], [0, np.pi / 2, 0])
    mid_branchId = planar_pruner.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1], [0, np.pi / 2, 0])
    bottom_branchId = planar_pruner.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 0.5], [0, np.pi / 2, 0])

    # Load the pruning points
    prune_point_0 = planar_pruner.load_urdf("sphere2.urdf", prune_point_0_pos, radius=radius)
    prune_point_1 = planar_pruner.load_urdf("sphere2.urdf", prune_point_1_pos, radius=radius)
    prune_point_2 = planar_pruner.load_urdf("sphere2.urdf", prune_point_2_pos, radius=radius)

    # Run the simulation
    for i in range(num_steps):
        # Example control: Set target position for the joints
        target_positions = [-0.5, 0.5, -0.5, -0.5, 0, 0]  # Example target positions for the three joints
        for j in range(3):
            p.setJointMotorControl2(
                bodyUniqueId=manipulatorId,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[j],
                force=500
            )
        p.stepSimulation()
        time.sleep(1 / sim_rate)  # Sleep to maintain a simulation step rate

    # Disconnect from the PyBullet physics engine
    p.disconnect()


if __name__ == "__main__":
    main()
