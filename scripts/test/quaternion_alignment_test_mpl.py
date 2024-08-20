import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def batch_line_orientations_quaternion(point1, points2):
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

def plot_line_with_orientation(pt1, pt2, quaternion, rand_pt=False):
    """
    Plot a 3D line starting from pt1 and pointing towards pt2, with orientation defined by a quaternion.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if rand_pt:
        rand_coord = np.random.uniform(-2, 2, 3)
        ax.scatter(rand_coord[0], rand_coord[1], rand_coord[2], color='b', label='Random Coord.', s=100)
    
    # Plot the original points
    ax.scatter(pt1[0], pt1[1], pt1[2], color='r', label='Point 1', s=100)
    ax.scatter(pt2[0][0], pt2[0][1], pt2[0][2], color='g', label='Point 2', s=100)
    
    # Compute the direction vector from the quaternion
    rotation = R.from_quat(quaternion[0])  # Use the first quaternion (since we have only 1 pair of points)
    direction_vector = rotation.apply([0, 0, 1])  # Apply the quaternion to the z-axis

    # Scale the direction vector to an arbitrary length (for visualization)
    line_end = pt2[0] + 0.1 * direction_vector

    # Plot the line from pt2 in the direction of the quaternion
    ax.plot([pt2[0][0], line_end[0]], [pt2[0][1], line_end[1]], [pt2[0][2], line_end[2]], label='Oriented Line', linewidth=5)

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


pt1 = [0.5, 0.7, 1.1]
pt2 = [[1, 0.6, 1.08263518]] 

# Compute the quaternion for the line orientation
quaternion = batch_line_orientations_quaternion(pt1, pt2)

# Plot the line based on the quaternion orientation
plot_line_with_orientation(pt1, pt2, quaternion, rand_pt=False)
