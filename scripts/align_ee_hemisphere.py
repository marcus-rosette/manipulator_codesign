import pybullet as p
import pybullet_data
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import time
from scipy.spatial.transform import Rotation as R


def sample_hemisphere_suface_pts(origin, radius, num_points=[30, 30]):
    """
    Generates points on the surface of a hemisphere that lies on the xz-plane protruding along the y-axis 
    """
    # Number of points along each dimension
    num_theta = num_points[0]
    num_phi = num_points[1]

    # Create a meshgrid for theta and phi
    theta = np.linspace(-np.pi, 0, num_theta)  # Azimuthal angle
    phi = np.linspace(0, np.pi, num_phi)  # Elevation angle (0 to pi/2 for hemisphere)

    theta, phi = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian and scale by radius
    x = radius * np.sin(phi) * np.cos(theta) + origin[0]
    y = radius * np.sin(phi) * np.sin(theta) + origin[1]
    z = radius * np.cos(phi) + origin[2] 

    # Flatten the coordinate arrays to create a list of points
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Combine the x, y, z arrays into a single array of points
    coordinates = np.vstack((x_flat, y_flat, z_flat)).T

    return np.unique(coordinates, axis=0) # Remove duplicate rows from the coordinates array

def end_effector_orientations(point1, points2):
    """
    Calculate the orientation quaternions of 3D lines passing through a single point1 and multiple points in points2.

    Parameters:
    - point1: Tuple or list of 3 coordinates (x1, y1, z1) for the first point.
    - points2: Array of shape (N, 3) where each row represents a point (x2, y2, z2) for the second point.

    Returns:
    - Array of shape (N, 4) where each row is a quaternion (x, y, z, w) representing the rotation from the z-axis to the line direction.
    """
    # Convert point1 to numpy array
    point1 = np.array(point1)
    
    # Compute the direction vectors for each point in points2
    direction_vectors = point1 - points2  # Subtract points2 from point1 (reverse direction)
    direction_vectors = direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]  # Normalize direction vectors
    
    # Define the reference direction (z-axis)
    reference_direction = np.array([0, 0, 1])
    
    quaternions = np.zeros((direction_vectors.shape[0], 4))  # Prepare an array for quaternions
    
    # Compute the quaternion for each direction vector
    for i, direction_vector in enumerate(direction_vectors):
        rotation = R.align_vectors([direction_vector], [reference_direction])[0]
        quaternions[i] = rotation.as_quat()  # scipy returns in (x, y, z, w) order by default
    
    return quaternions

def test():
    # Start the PyBullet 
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load default data
    
    robot_id = p.loadURDF("./urdf/ur5e/ur5e_cart.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

    num_joints = p.getNumJoints(robot_id)

    # Get the end-effector link index 
    end_effector_index = num_joints - 3

    controllable_joint_idx = [
        p.getJointInfo(robot_id, joint)[0]
        for joint in range(num_joints)
        if p.getJointInfo(robot_id, joint)[2] in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}
    ]

    # Target position and orientation for the end-effector
    look_at_point = [0, 0.8, 1.2]    # Point where the end-effector should face

    # Visualize the look_at_point as a sphere
    look_at_sphere = p.loadURDF("sphere2.urdf", look_at_point, globalScaling=0.05, useFixedBase=True)
    p.changeVisualShape(look_at_sphere, -1, rgbaColor=[0, 1, 0, 1]) 

    look_at_point_offset = 0.1
    hemisphere_center = np.copy(look_at_point)
    hemisphere_center[1] -= look_at_point_offset
    num_points = [10, 10]

    hemisphere_pts = sample_hemisphere_suface_pts(hemisphere_center, 0.1, num_points)
    hemisphere_oris = end_effector_orientations(look_at_point, hemisphere_pts)

    poi = 30
    target_position = hemisphere_pts[poi]
    target_orientation = hemisphere_oris[poi]

    # Add a debug line between the two points
    line_id = p.addUserDebugLine(lineFromXYZ=target_position,
                                lineToXYZ=look_at_point,
                                lineColorRGB=[1, 0, 0],  # Red color
                                lineWidth=4)


    poi_id = p.loadURDF("sphere2.urdf", target_position, globalScaling=0.05, useFixedBase=True)
    p.changeVisualShape(poi_id, -1, rgbaColor=[1, 0, 0, 1]) 

    # Use PyBullet's inverse kinematics to compute joint angles
    joint_angles = p.calculateInverseKinematics(
        robot_id,
        end_effector_index,
        target_position,
        target_orientation
    )

    # Iterate over the joints and set their positions
    for i, joint_idx in enumerate(controllable_joint_idx):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_angles[i]
        )

    while True:
        p.stepSimulation()


if __name__ == '__main__':
    test()
