import numpy as np
import open3d as o3d


def simple_point_cloud_viz():
    # Load the Bunny mesh and convert it to a point cloud
    bunny_mesh = o3d.data.BunnyMesh().path
    bunny = o3d.io.read_triangle_mesh(bunny_mesh)
    bunny_pcd = bunny.sample_points_poisson_disk(1000) # Set the number of uniformly sampled points in the point cloud

    voxel_size = 0.01

    # Apply voxel downsampling (set the size of each voxel)
    bunny_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(bunny_pcd, voxel_size=voxel_size)

    # Extract the voxel coordinates
    voxel_coordinates = [voxel.grid_index for voxel in bunny_voxel_grid.get_voxels()]

    # Convert voxel grid indices to 3D coordinates
    voxel_centers = [bunny_voxel_grid.get_voxel_center_coordinate(v) for v in voxel_coordinates]

    # Print the first few voxel centers to verify
    for i, center in enumerate(voxel_centers[:5]):
        print(f"Voxel {i+1}: Center Coordinate: {center}")

    o3d.visualization.draw_geometries([bunny_pcd])


if __name__ == '__main__':
    simple_point_cloud_viz()