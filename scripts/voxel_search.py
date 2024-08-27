import numpy as np


def path_to_closest_voxel(target_point, voxel_centers, paths):
    """ Find the path to a voxel that the target point is closest to 

    Args:
        target_point (float list): target 3D coordinate
        voxel_centers (float list): list of voxel center coordinates to search through
        paths (float list): list of joint trajectories associated with the voxel centers

    Returns:
        path: the path to the voxel the target point is closes to
    """
    # Calculate distances
    distances = np.linalg.norm(voxel_centers - target_point, axis=1)
    
    # Find the index of the closest voxel
    closest_voxel_index = np.argmin(distances)

    print(f'Distance error: {distances[closest_voxel_index]}')

    # Get the associated path to closest voxel
    path = paths[:, :, closest_voxel_index]
    
    return path


if __name__ == "__main__":
    voxel_data = np.loadtxt('./data/voxel_data_parallelepiped.csv')
    voxel_centers = voxel_data[:, :3]
    voxel_indices = voxel_data[:, 3:]

    paths = np.load('./data/voxel_paths_parallelepiped.npy')

    target_point = np.array([0.1, 0.6, 1.2]) 

    path = path_to_closest_voxel(target_point, voxel_centers, paths)




