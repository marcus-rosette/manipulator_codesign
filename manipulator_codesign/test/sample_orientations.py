import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from scipy.spatial.transform import Rotation as R


def generate_northern_hemisphere_orientations(num_orientations):
    """
    Generate orientations with Z-axes pointing over the northern hemisphere.
    
    Args:
        num_orientations (int): Number of orientations to generate.
    
    Returns:
        List[np.ndarray]: List of 3x3 rotation matrices.
    """
    quaternions = []
    golden_ratio = (1 + 5 ** 0.5) / 2  # φ ~ 1.618

    for i in range(num_orientations):
        # Fibonacci spiral on hemisphere
        theta = 2 * np.pi * i / golden_ratio
        z = 1 - (i + 0.5) / num_orientations  # [0,1)
        r = np.sqrt(1 - z * z)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Hemisphere point (on surface)
        hemisphere_pt = np.array([x, y, z])

        # Inward-facing Z-axis
        z_axis = -hemisphere_pt / np.linalg.norm(hemisphere_pt)

        # Construct orthonormal basis
        up = np.array([0, 0, 1])
        if np.allclose(z_axis, up) or np.allclose(z_axis, -up):
            up = np.array([0, 1, 0])

        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Construct rotation matrix and convert to quaternion
        R_mat = np.column_stack((x_axis, y_axis, z_axis))
        quat = R.from_matrix(R_mat).as_quat()  # (x, y, z, w)
        quaternions.append(quat)

    return quaternions

def plot_orientation_z_axes(orientations, axis_length=0.1):
    """
    Plot Z-axes of given orientation matrices as vectors on the unit hemisphere.
    
    Args:
        orientations (List[np.ndarray]): List of 3x3 rotation matrices.
        axis_length (float): Length of the axis vectors for visualization.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw hemisphere surface (for reference)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, linewidth=0)

    # Plot each Z-axis as an arrow from origin
    for R_mat in orientations:
        z_axis = R_mat[:, 2]  # third column is Z-axis
        ax.quiver(0, 0, 0, 
                  z_axis[0], z_axis[1], z_axis[2], 
                  length=axis_length, normalize=True, color='r')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Z-Axes of Generated Orientations (Northern Hemisphere)")
    plt.tight_layout()
    # plt.show()
    plt.savefig("hemisphere_orientations.png", dpi=300)


if __name__ == "__main__":
    num_orientations = 50
    orientations = generate_northern_hemisphere_orientations(num_orientations)
    plot_orientation_z_axes(orientations, axis_length=1)
    
    # Example: Print the first orientation matrix
    print("First orientation matrix (Z-axis):")
    print(orientations[0])