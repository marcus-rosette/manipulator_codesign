import os
import numpy as np
from scipy.spatial.transform import Rotation
from manipulator_codesign.genetic_algorithm import GeneticAlgorithm


current_working_directory = os.getcwd()
save_urdf_dir = os.path.join(current_working_directory, 'manipulator_codesign/urdf/robots/')

# Bool to sample target end-effector poses from a list of predefined poses
# If False, target end-effector positions will be used (no orientation)
# If True, target end-effector positions and orientations will be used
sample_target_poses = False


# Example Target End-Effector Position
target_positions = [
    [1.0, 1.0, 1.0],
    [0.5, 1.5, 1.2],
    [-0.5, 1.0, 0.8],
    [0.0, 1.0, 1.0],
    [-1.0, 1.5, 1.2],
    [-0.5, 0.5, 0.1],
    [0.5, 0.5, 1.0],
    [1.2, -0.5, 0.9],
    [-1.2, 0.8, 1.1],
    [0.3, -1.0, 1.3],
    [-0.7, -1.2, 0.7],
    [0.8, 1.3, 1.4],
    [-1.1, -0.8, 0.6],
    [0.6, -0.6, 1.5],
    [-0.3, 0.7, 1.2]
]

target_orientations = [
    Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [-90, 90, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
    Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat()
]

if sample_target_poses:
    target_poses = list(zip(target_positions, target_orientations))
else:
    target_poses = np.array(target_positions)

ga_pyb = GeneticAlgorithm(target_poses, save_urdf_dir=save_urdf_dir, backend='pybullet', 
                            population_size=20, generations=10, renders=False, max_num_joints=7,
                            mutation_rate=0.4, crossover_rate=0.6)

best_chain, total_generated, total_iters = ga_pyb.run()

print(f"\nTotal Kinematic Chains Generated: {total_generated}")
print(f"Total Iterations: {total_iters}")
best_chain.describe()