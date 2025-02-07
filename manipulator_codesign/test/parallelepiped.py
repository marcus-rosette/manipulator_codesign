import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

def generate_voxels(vertices, resolution):
    """ Generate discrete 3D coordinates of the volume that makes up the mesh """
    # Get the axis-aligned bounding box of the parallelepiped
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    # Generate voxel grid coordinates
    x = np.arange(min_coords[0], max_coords[0] + resolution, resolution)
    y = np.arange(min_coords[1], max_coords[1] + resolution, resolution)
    z = np.arange(min_coords[2], max_coords[2] + resolution, resolution)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    voxel_coords = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    # Filter voxels inside the parallelepiped
    inside_voxels = np.array([coord for coord in voxel_coords if is_point_in_parallelepiped(coord, vertices)])

    return inside_voxels

def is_point_in_parallelepiped(point, vertices):
    """ Check if a point is inside the parallelepiped defined by its vertices """
    # Create a convex hull from the vertices
    hull = ConvexHull(vertices)
    # Check if the point is inside the convex hull
    new_point = np.append(point, 1)  # Append 1 for homogeneous coordinates
    return np.all(np.dot(hull.equations[:, :-1], new_point[:-1]) <= -hull.equations[:, -1])

def plot_voxels_and_surface(voxels, vertices):
    """ Plot the 3D scatter plot of voxel coordinates and surface points """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot for voxels
    ax.scatter(voxels[:, 0], voxels[:, 1], voxels[:, 2], c='b', marker='o', label='Voxels')
    
    # Scatter plot for surface points
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', marker='^', label='Surface Points')
    
    ax.set_xlabel('X axis (Width)')
    ax.set_ylabel('Y axis (Depth)')
    ax.set_zlabel('Z axis (Height)')
    ax.set_title('3D Scatter Plot of Voxels and Surface Points')
    ax.legend()
    # Set axis limits
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((0, 2))
    
    plt.show()

# Example usage
height = 2.0
width = 1.0
depth = 0.2
theta = np.deg2rad(18.435)  # Skew angle in radians
resolution = 0.1  # Voxel size

# Define vertices of the parallelepiped
vertices = np.array([
        [0, 0, 0],  # v0
        [2*width, 0, 0],  # v1
        [2*width, 2*depth, 0],  # v2
        [0, 2*depth, 0],  # v3
        [0, height*np.tan(theta), height],  # v4
        [2*width, height*np.tan(theta), height],  # v5
        [2*width, 2*depth + height*np.tan(theta), height],  # v6
        [0, 2*depth + height*np.tan(theta), height]  # v7
    ])

voxels = generate_voxels(vertices, resolution)

# # Define the rotation matrix around the x-axis
# R_y = R.from_euler('y', np.pi).as_matrix()
# rotated_voxels = np.dot(vertices, R_y.T)
print(max(voxels[:, 1]))

voxels[:, 0] -= max(voxels[:, 0]) / 2
voxels[:, 1] *= -1
voxels[:, 1] -= min(voxels[:, 1]) / 2

vertices[:, 0] -= max(vertices[:, 0]) / 2
vertices[:, 1] *= -1
vertices[:, 1] -= min(vertices[:, 1]) / 2

print(f'Number of voxels: {len(voxels)}')
print(voxels)

# Plot the voxels and surface points
plot_voxels_and_surface(voxels, vertices)
plt.axis('equal')
