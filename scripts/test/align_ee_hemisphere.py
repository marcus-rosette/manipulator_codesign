import pybullet as p
import pybullet_data
import numpy as np
import time
import sys
import os

# Get the file from the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from sample_approach_points import SamplePoints


def test():
    # Start the PyBullet 
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load default data
    
    robot_id = p.loadURDF("./urdf/ur5e/ur5e_cart.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

    num_joints = p.getNumJoints(robot_id)

    # Get the end-effector link index 
    end_effector_index = num_joints - 3

    controllable_joint_idx = [
        p.getJointInfo(robot_id, joint)[0]
        for joint in range(num_joints)
        if p.getJointInfo(robot_id, joint)[2] in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}
    ]

    # Set the home position
    home_configuration = [-3 * np.pi/4, -np.pi/3, 2 * np.pi/3, 2 * np.pi/3, -np.pi/2, 0]  # Example home configuration
    for i, joint_idx in enumerate(controllable_joint_idx):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=home_configuration[i]
        )

    # Target position and orientation for the end-effector
    look_at_point = [0, 0.8, 1.2]    # Point where the end-effector should face
    look_at_point_offset = 0.1
    num_points = [10, 10]

    point_sampler = SamplePoints(p, planar=False)

    # Visualize the look_at_point as a sphere
    look_at_sphere = p.loadURDF("sphere2.urdf", look_at_point, globalScaling=0.05, useFixedBase=True)
    p.changeVisualShape(look_at_sphere, -1, rgbaColor=[0, 1, 0, 1]) 

    hemisphere_pts = point_sampler.sample_hemisphere_suface_pts(look_at_point, look_at_point_offset, 0.1, num_points)
    hemisphere_oris = point_sampler.hemisphere_orientations(look_at_point, hemisphere_pts)

    # Iterate through position and orientation pairs
    for target_position, target_orientation in zip(hemisphere_pts, hemisphere_oris):
        # Add a debug line between the two points
        line_id = p.addUserDebugLine(lineFromXYZ=target_position,
                                    lineToXYZ=look_at_point,
                                    lineColorRGB=[1, 0, 0],  # Red color
                                    lineWidth=2)


        poi_id = p.loadURDF("sphere2.urdf", target_position, globalScaling=0.02, useFixedBase=True)
        p.changeVisualShape(poi_id, -1, rgbaColor=[1, 0, 0, 1]) 

        # Use PyBullet's inverse kinematics to compute joint angles
        joint_angles = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            target_position,
            target_orientation,
            residualThreshold=0.01
        )

        # Iterate over the joints and set their positions
        for i, joint_idx in enumerate(controllable_joint_idx):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i]
            )

        # Step simulation and render
        for _ in range(240):  # Adjust number of simulation steps as needed
            p.stepSimulation()
            time.sleep(1./240.)  # Sleep to match real-time

        # Pause to view the result before moving to the next target
        input("Press Enter to view the next target...")

    # while True:
    #     p.stepSimulation()


if __name__ == '__main__':
    test()
