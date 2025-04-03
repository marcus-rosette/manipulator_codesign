import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from manipulator_codesign.genetic_algorithm import GeneticAlgorithm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm for Kinematic Chain Optimization")
    
    parser.add_argument("-p", "--population_size", type=int, default=30, help="Size of the population. Defaults to 100")
    parser.add_argument("-g", "--generations", type=int, default=50, help="Number of generations. Defaults to 50")
    parser.add_argument("-j", "--max_num_joints", type=int, default=7, help="Maximum number of joints. Defaults to 7")
    parser.add_argument("-m", "--mutation_rate", type=float, default=0.3, help="Mutation rate. Defaults to 0.3")
    parser.add_argument("-c", "--crossover_rate", type=float, default=0.7, help="Crossover rate. Defaults to 0.7")
    parser.add_argument("-r", "--renders", action="store_true", help="Enable rendering in PyBullet simulation. Defaults to False")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    current_working_directory = os.getcwd()
    save_urdf_dir = os.path.join(current_working_directory, 'manipulator_codesign/urdf/robots/')

    # Bool to sample target end-effector poses from a list of predefined poses
    sample_target_poses = False

    target_positions = np.random.uniform(low=[-2.0, -2.0, 0], high=[2.0, 2.0, 2.0], size=(20, 3)).tolist()
    
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

    ga_pyb = GeneticAlgorithm(
        target_poses,
        save_urdf_dir=save_urdf_dir,
        backend='pybullet',
        population_size=args.population_size,
        generations=args.generations,
        max_num_joints=args.max_num_joints,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        renders=args.renders
    )

    best_chain, total_generated, total_iters = ga_pyb.run()

    print(f"\nTotal Kinematic Chains Generated: {total_generated}")
    print(f"Total Iterations: {total_iters}")
    best_chain.describe()
