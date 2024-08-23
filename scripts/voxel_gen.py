import numpy as np
import open3d as o3d


def filter_isolated_points(point_cloud, distance_threshold=0.1):
    # Convert point cloud to a numpy array for easy manipulation
    points = np.asarray(point_cloud.points)
    
    # Use a KDTree for efficient neighbor search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    
    # List to hold indices of points that should be kept
    valid_indices = []
    
    for i in range(len(points)):
        # Find neighbors within the specified radius
        [k, idx, _] = kdtree.search_radius_vector_3d(point_cloud.points[i], distance_threshold)
        
        # If more than 1 point is found (including the point itself), keep this point
        if k > 1:
            valid_indices.append(i)
    
    # Filter points and colors based on valid indices
    filtered_points = points[valid_indices]
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        filtered_colors = colors[valid_indices]
    
    # Create a new point cloud with the filtered points (and colors if available)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    if point_cloud.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    return filtered_pcd

def load_pc_and_downsample(file_path, z_threshold=0, neighbor_threshold=0.1, voxel_size=0.1, vis=True):
    pcd = o3d.io.read_point_cloud(file_path)

    points = np.asarray(pcd.points) / 1000.0 # Convert from mm to m
    colors = np.asarray(pcd.colors)

    # Invert axes (if neccesary)
    points[:, 0] *= -1  # Invert the x-axis
    points[:, 1] *= -1  # Invert the y-axis
    # points[:, 2] *= -1  # Invert the z-axis

    # Filter points and corresponding colors based on the z-coordinate (maintaining original colors)
    filtered_indices = points[:, 2] <= z_threshold
    filtered_points = points[filtered_indices]
    filtered_colors = colors[filtered_indices]

    # Create a new point cloud with the filtered points and colors
    z_filtered_pcd = o3d.geometry.PointCloud()
    z_filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    z_filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)  # Preserve the colors

    # Apply voxel downsampling (set the size of each voxel)
    downpcd = z_filtered_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Filter out any isolated neighboring points
    filtered_pcd = filter_isolated_points(downpcd, distance_threshold=neighbor_threshold)

    # Create a voxel grid from the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(filtered_pcd, voxel_size=voxel_size)

    # Get the voxel data
    voxels = voxel_grid.get_voxels()
    
    # Extract voxel centers
    voxel_centers = [voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels]

    print(f'Original number of points: {len(points)}')
    print(f'Downsampled number of voxels: {len(voxel_centers)}')

    # Visualize the voxel grid (shows the voxel boundaries)
    if vis:
        o3d.visualization.draw_geometries([voxel_grid])

    return voxel_centers


if __name__ == '__main__':
    file_path = "./point_clouds/pointcloud_2.ply"
    z_threshold = 1.5 # meters
    neighbor_threshold = 0.1 # meters
    voxel_size = 0.1 # meters

    voxel_centers = load_pc_and_downsample(file_path, z_threshold, neighbor_threshold, voxel_size, vis=True)

    np.savetxt('./data/voxel_centers2.csv', voxel_centers)
