import pybullet as p
import time
import pybullet_data
import numpy as np


class PlanarPruner:
    def __init__(self):
        # Connect to the PyBullet physics engine
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
    
    def prune_arc(self, prune_point, radius, allowance_angle, num_points, x_ori_default=0.0, z_ori_default=np.pi/2):
        # Define theta as a descrete array
        theta = np.linspace(-allowance_angle, allowance_angle, num_points)

        # Set up arc length coordinate
        x = np.full_like(theta, prune_point[0])  # x-coordinate remains constant
        y = - radius * np.cos(theta) + prune_point[1] # multiply by a negative to mirror on other side of axis
        z = radius * np.sin(theta) + prune_point[2] 

        # Calculate orientation angles
        arc_angles = np.arctan2(prune_point[2] - z, prune_point[1] - y)

        arc_coords = np.vstack((x, y, z))

        goal_coords = np.zeros((num_points, 3)) # 3 for x, y, z
        goal_orientations = np.zeros((num_points, 4)) # 4 for quaternion
        for i in range(num_points):
            goal_coords[i] = [arc_coords[0][i], arc_coords[1][i], arc_coords[2][i]]
            # goal_orientations[i] = p.getQuaternionFromEuler([x_ori_default, arc_angles[i] + np.pi + 1.57, z_ori_default])
            goal_orientations[i] = p.getQuaternionFromEuler([x_ori_default, arc_angles[i] + np.pi/2, z_ori_default])

        return goal_coords, goal_orientations
    
    def calculate_manipulability(self, robot, ee_index, joint_positions, end_effector_pos, planar=True):
        joint_positions = list(joint_positions)
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
        
        # Check for singularity
        if np.linalg.matrix_rank(jacobian) < 3:
            print("\nSingularity detected: Manipulability is zero.")
            return 0.0

        if planar:
            jac_t = np.array(jac_t)[1:3]
            jac_r = np.array(jac_r)[0]
            jacobian = np.vstack((jac_t, jac_r))

        return np.sqrt(np.linalg.det(jacobian @ jacobian.T))


def main():
    planar_pruner = PlanarPruner()

    # Load the manipulator
    manipulatorId = planar_pruner.load_urdf("./urdf/rrrp_manipulator.urdf", [0, 0, 0])

    # Set initial joint positions
    num_joints = p.getNumJoints(manipulatorId)

    target_ee_pos = [0.5, 0.95, 1.1]
    
    points, oris = planar_pruner.prune_arc(target_ee_pos, 0.1, np.deg2rad(30), 10)

    # target_ori = [-0.06162842,  0.06162842,  0.70441603,  0.70441603]
    target_ori = p.getQuaternionFromEuler([0, 1.57, 1.57])

    # Set target joint position 
    target_positions = [-0.5, 0.5, 0.5, -0.5] 
    # target_positions = [-1.57, 0.5, -1, 1, -1.5, 0]

    poi = 1

    # target_positions = np.array(p.calculateInverseKinematics(manipulatorId, num_joints - 1, target_ee_pos, target_ori))
    # target_positions = np.array(p.calculateInverseKinematics(manipulatorId, num_joints - 1, points[poi], oris[poi]))
    # print(target_positions)
    # print(target_positions)

    ee_index = p.getNumJoints(manipulatorId) - 1

    # Iterate over the joints and set their positions
    for i in range(num_joints):
        joint_info = p.getJointInfo(manipulatorId, i)
        joint_type = joint_info[2]

        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            p.setJointMotorControl2(
                bodyIndex=manipulatorId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[i]
            )

    # Wait until the manipulator reaches the target positions
    tolerance = 0.45  # Position tolerance
    while True:
        joint_states = [p.getJointState(manipulatorId, i)[0] for i in range(num_joints)]
        if all(abs(joint_states[i] - target_positions[i]) < tolerance for i in range(len(target_positions))):
            break
        p.stepSimulation()
        time.sleep(0.01)

    final_end_effector_state = p.getLinkState(manipulatorId, ee_index)
    final_end_effector_pos = np.array(final_end_effector_state[0])
    final_end_effector_orientation = np.array(final_end_effector_state[1])

    # Plot manipulator manipulability after reaching target positions
    manipulability = planar_pruner.calculate_manipulability(manipulatorId, ee_index, target_positions, np.array(final_end_effector_pos), planar=True)
    print(manipulability)

    # Run the simulation for visualization
    p.setRealTimeSimulation(1)

    # Keep the simulation running
    while True:
        p.stepSimulation()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
