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
        List[np.ndarray]: quternion list of shape (num_orientations, 4) in (x, y, z, w) format.
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

def downsample_quaternions_facing_robot(quaternions, target_position, robot_base_position, threshold=0.0, percent_toward=0.5):
    """
    Filters and downsamples quaternions to retain a specified percentage of those whose Z-axes point toward the robot base position.

    Args:
        quaternions (List[np.ndarray]): List of quaternions in (x, y, z, w) format.
        target_position (np.ndarray): 3D position the orientations point toward (hemisphere center).
        robot_base_position (np.ndarray): 3D position of the robot base.
        threshold (float): Dot product threshold for filtering; higher means stricter culling.
        percent_toward (float): Fraction (0-1) of 'toward' orientations to keep.

    Returns:
        List[np.ndarray]: Filtered and downsampled list of quaternions.
    """
    toward = []
    away = []

    view_vector = np.array(robot_base_position) - np.array(target_position)
    view_vector /= np.linalg.norm(view_vector)

    for quat in quaternions:
        z_axis = R.from_quat(quat).as_matrix()[:, 2]
        alignment = np.dot(z_axis, view_vector)

        if alignment > threshold:
            toward.append(quat)
        elif alignment < -threshold:
            away.append(quat)

    # Downsample 'toward' orientations based on percent_toward
    num_toward_to_keep = int(len(toward) * percent_toward)
    rng = np.random.default_rng(42)  # For reproducibility
    if num_toward_to_keep > 0:
        toward_downsampled = rng.choice(toward, size=num_toward_to_keep, replace=False).tolist()
    else:
        toward_downsampled = []

    return toward_downsampled + away

def sample_collision_free_poses(robot, collision_objects, robot_position=np.array([0.0, 0.0, 0.0]), target_points=None, num_orientations=100):
    orientations = generate_northern_hemisphere_orientations(num_orientations)

    target_poses = []
    for i, target_pt in enumerate(target_points):
        # filtered_orientations = orientations.copy() # Start with all orientations
        # filtered_orientations = downsample_quaternions_facing_robot(
        #     orientations, target_pt, robot_base_position=robot_position, threshold=0.0, percent_toward=0.2
        # )

        # # Sort orientations by alignment (dot product) of Z-axis with vector toward robot base
        # view_vector = np.array(robot_position) - np.array(target_pt)
        # view_vector /= np.linalg.norm(view_vector)
        # filtered_orientations.sort(
        #     key=lambda quat: -np.dot(R.from_quat(quat).as_matrix()[:, 2], view_vector),
        #     reverse=True
        # )

        for orientation in orientations:
            target_pose = (target_pt, orientation)
            joint_config = robot.inverse_kinematics(target_pose, pos_tol=0.01, rest_config=robot.home_config, max_iter=1000, num_resample=5)

            # Check if the joint configuration is collision-free
            if joint_config is not None:
                robot.reset_joint_positions(joint_config)

                if robot.collision_check(robot.robotId, collision_objects):
                    orientation = np.array((1, 0, 0, 0)) # Default orientation if collision occurs (considered as no orientation)
                target_poses.append((target_pt, orientation))
                break
    return target_poses

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

    # Step 1: Generate orientations (as quaternions)
    quaternions = generate_northern_hemisphere_orientations(num_orientations)

    # Step 2: Define positions
    target_position = np.array([0.0, 0.5, 0.5])
    robot_base_position = np.array([0.0, 0.0, 0.0])  # Behind the hemisphere

    # Step 3: Downsample orientations that face the robot
    filtered_quats = downsample_quaternions_facing_robot(
        quaternions,
        target_position=target_position,
        robot_base_position=robot_base_position,
        threshold=0.0  # Optional: increase for stricter filtering
    )

    # Step 4: Convert filtered quaternions to rotation matrices
    filtered_rot_mats = [R.from_quat(q).as_matrix() for q in filtered_quats]

    # Step 5: Plot the downsampled orientations
    plot_orientation_z_axes(quaternions, axis_length=1)

    # Optional: Print info
    print(f"Filtered from {len(quaternions)} to {len(filtered_quats)} orientations")