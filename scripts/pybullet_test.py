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
    
    def calculate_manipulability(self, robot, ee_index, joint_positions, end_effector_pos):
        zero_vec = [0.0] * len(joint_positions)
        jac_t, jac_r = p.calculateJacobian(robot, ee_index, [0, 0, 0], joint_positions, zero_vec, zero_vec)
        jacobian = np.vstack((jac_t, jac_r))

        # Visualization of the Jacobian columns
        num_columns = jacobian.shape[1]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]  # Different colors for each column
        for i in range(num_columns):

            vector = jacobian[:, i]
            start_point = end_effector_pos
            end_point = start_point + 0.3 * vector[:3]  # Scale the vector for better visualization
            p.addUserDebugLine(start_point, end_point, colors[i % len(colors)], 2)
            print(colors[i % len(colors)])
        
        # Check for singularity
        if np.linalg.matrix_rank(jacobian) < 3:
            print("\nSingularity detected: Manipulability is zero.")
            return 0.0
        
        manipulability_measure = np.sqrt(np.linalg.det(jacobian @ jacobian.T))
        return manipulability_measure


def main():
    planar_pruner = PlanarPruner()

    # Define the starting position of the branches
    start_x = 0.5
    start_y = 1

    # Define the starting position of the pruning points
    prune_point_0_pos = [start_x, start_y, 1.55]
    prune_point_1_pos = [start_x, start_y - 0.05, 1.05]
    prune_point_2_pos = [start_x, start_y + 0.05, 0.55]
    radius = 0.05

    # Load the manipulator
    manipulatorId = planar_pruner.load_urdf("./urdf/rrr_manipulator.urdf", [start_x, 0, 0])

    # Set initial joint positions
    initial_positions = [0.0, 0.0, 0.0]  # Example initial positions for the three joints
    num_joints = p.getNumJoints(manipulatorId)
    for i in range(3):
        p.resetJointState(manipulatorId, i, initial_positions[i])

    # # Load the branches
    # leader_branchId = planar_pruner.load_urdf("./urdf/leader_branch.urdf", [0, start_y, 1.6/2])
    # top_branchId = planar_pruner.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1.5], [0, np.pi / 2, 0])
    # mid_branchId = planar_pruner.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1], [0, np.pi / 2, 0])
    # bottom_branchId = planar_pruner.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 0.5], [0, np.pi / 2, 0])

    # # Load the pruning points
    # prune_point_0 = planar_pruner.load_urdf("sphere2.urdf", prune_point_0_pos, radius=radius)
    # prune_point_1 = planar_pruner.load_urdf("sphere2.urdf", prune_point_1_pos, radius=radius)
    # prune_point_2 = planar_pruner.load_urdf("sphere2.urdf", prune_point_2_pos, radius=radius)

    # Example control: Set target position for the joints
    target_positions = [-0.3, -0.5, -0.9]  # Example target positions for the three joints

    ee_index = p.getNumJoints(manipulatorId) - 1

    # Iterate over the joints and set their positions
    for i in range(num_joints):
        joint_info = p.getJointInfo(manipulatorId, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]

        if joint_type == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                bodyIndex=manipulatorId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[i]
            )

    # Wait until the manipulator reaches the target positions
    tolerance = 0.01  # Position tolerance
    while True:
        joint_states = [p.getJointState(manipulatorId, i)[0] for i in range(num_joints)]
        if all(abs(joint_states[i] - target_positions[i]) < tolerance for i in range(len(target_positions))):
            break
        p.stepSimulation()
        time.sleep(0.01)

    # Print the name of the link associated with ee_index
    link_info = p.getJointInfo(manipulatorId, ee_index)
    link_name = link_info[12].decode('utf-8')  # Link name is at index 12
    print(f"End-Effector Link Name: {link_name}")

    final_end_effector_state = p.getLinkState(manipulatorId, ee_index)
    final_end_effector_pos = np.array(final_end_effector_state[0])
    final_end_effector_orientation = np.array(final_end_effector_state[1])

    # Plot manipulator manipulability after reaching target positions
    print(planar_pruner.calculate_manipulability(manipulatorId, ee_index, target_positions, np.array(final_end_effector_pos)))

    # Run the simulation for visualization
    p.setRealTimeSimulation(1)

    # Keep the simulation running
    while True:
        p.stepSimulation()
        time.sleep(0.01)

    # # Disconnect from the PyBullet physics engine
    # p.disconnect()



if __name__ == "__main__":
    main()
