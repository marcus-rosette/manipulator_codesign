import numpy as np
import random
import ray
from manipulator_codesign.kinematic_chain import KinematicChainRTB, KinematicChainPyBullet

# Initialize Ray
ray.init(ignore_reinit_error=True)

@ray.remote
def evaluate_fitness_remote(ga, chain):
    """
    Ray remote function to compute fitness.
    The GA instance 'ga' is passed so that the fitness function can access normalization factors.
    It is assumed that chain has precomputed attributes:
      - mean_pose_error
      - mean_torque
      - conditioning_index
    """
    return ga.fitness(chain)

class GeneticAlgorithm:
    def __init__(self, target_positions, backend='pybullet', save_urdf_dir=None, 
                 max_num_joints=5, link_len_bounds=(0.05, 1.0), sort_pop_low=True,
                 population_size=20, generations=100, mutation_rate=0.3, 
                 crossover_rate=0.7, joint_penalty_weight=0.1, renders=False):
        self.target_positions = target_positions
        self.backend = backend
        self.save_urdf_dir = save_urdf_dir
        self.max_num_joints = max_num_joints
        self.link_lengths_bounds = link_len_bounds
        self.sort_pop_low = sort_pop_low
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate  
        self.crossover_rate = crossover_rate
        self.joint_penalty_weight = joint_penalty_weight

        # For dynamic normalization in fitness evaluation
        self.population_pose_errors = []
        self.population_torques = []

        if self.backend == 'pybullet':
            from manipulator_codesign.pyb_utils import PybUtils
            from manipulator_codesign.load_objects import LoadObjects
            self.pyb = PybUtils(renders=renders)
            self.object_loader = LoadObjects(self.pyb.con)
    
    def _chain_factory(self, num_joints, joint_types, joint_axes, link_lengths):
        if self.backend == 'pybullet':
            return KinematicChainPyBullet(self.pyb.con, num_joints, joint_types, joint_axes, link_lengths,
                                          save_urdf_dir=self.save_urdf_dir)
        else:
            return KinematicChainRTB(num_joints, joint_types, joint_axes, link_lengths,
                                     save_urdf_dir=self.save_urdf_dir)
    
    def generate_random_chain(self):
        num_joints = random.randint(2, self.max_num_joints)
        joint_types = [random.choice([0, 1]) for _ in range(num_joints)]
        joint_axes = [random.choice(['x', 'y', 'z']) for _ in range(num_joints)]
        link_lengths = [random.uniform(*self.link_lengths_bounds) for _ in range(num_joints)]
        return self._chain_factory(num_joints, joint_types, joint_axes, link_lengths)
    
    def fitness(self, chain):
        """
        Compute fitness using precomputed metrics.
        This function only uses the chain’s precomputed attributes:
         - mean_pose_error
         - mean_torque
         - conditioning_index
        """
        alpha = 5    # Pose error weight
        beta = 0.1   # Torque weight
        gamma = 2.0  # Conditioning index weight

        pose_error_norm = max(1e-6, np.max(self.population_pose_errors))
        torque_norm = max(1e-6, np.max(self.population_torques))

        normalized_pose_error = chain.mean_pose_error / pose_error_norm
        normalized_torque_penalty = chain.mean_torque / torque_norm

        return alpha * normalized_pose_error + beta * normalized_torque_penalty - gamma * chain.conditioning_index

    def evaluate_population_fitness(self, population):
        """
        Evaluate fitness for each chain in the population using Ray.
        """
        futures = [evaluate_fitness_remote.remote(self, chain) for chain in population]
        return ray.get(futures)
    
    def crossover(self, parent1, parent2):
        if parent1.num_joints == parent2.num_joints:
            child_num_joints = parent1.num_joints
        else:
            child_num_joints = random.choice([parent1.num_joints, parent2.num_joints])
        
        child_joint_types = []
        child_joint_axes = []
        child_link_lengths = []
        for i in range(child_num_joints):
            gene_from_parent1 = i < parent1.num_joints
            gene_from_parent2 = i < parent2.num_joints
            if gene_from_parent1 and gene_from_parent2:
                joint_type = random.choice([parent1.joint_types[i], parent2.joint_types[i]])
                joint_axis = random.choice([parent1.joint_axes[i], parent2.joint_axes[i]])
                link_length = (parent1.link_lengths[i] + parent2.link_lengths[i]) / 2
            elif gene_from_parent1:
                joint_type = parent1.joint_types[i]
                joint_axis = parent1.joint_axes[i]
                link_length = parent1.link_lengths[i]
            else:
                joint_type = parent2.joint_types[i]
                joint_axis = parent2.joint_axes[i]
                link_length = parent2.link_lengths[i]
            child_joint_types.append(joint_type)
            child_joint_axes.append(joint_axis)
            child_link_lengths.append(link_length)
        return self._chain_factory(child_num_joints, child_joint_types, child_joint_axes, child_link_lengths)
    
    def mutate(self, chain):
        new_num_joints = chain.num_joints
        new_joint_types = chain.joint_types.copy()
        new_joint_axes = chain.joint_axes.copy()
        new_link_lengths = chain.link_lengths.copy()
        for i in range(new_num_joints):
            if random.random() < self.mutation_rate:
                delta = random.gauss(0, 0.05)
                new_link_lengths[i] = np.clip(new_link_lengths[i] + delta, *self.link_lengths_bounds)
        if random.random() < (self.mutation_rate / 2):
            idx = random.randint(0, new_num_joints - 1)
            new_joint_types[idx] = 1 - new_joint_types[idx]
        if random.random() < (self.mutation_rate / 2):
            idx = random.randint(0, new_num_joints - 1)
            new_joint_axes[idx] = random.choice(['x', 'y', 'z'])
        if random.random() < (self.mutation_rate / 4) and new_num_joints > 2:
            idx = random.randint(0, new_num_joints - 1)
            del new_joint_types[idx]
            del new_joint_axes[idx]
            del new_link_lengths[idx]
            new_num_joints -= 1
        if random.random() < (self.mutation_rate / 4) and new_num_joints < self.max_num_joints:
            new_joint_types.append(random.choice([0, 1]))
            new_joint_axes.append(random.choice(['x', 'y', 'z']))
            new_link_lengths.append(random.uniform(*self.link_lengths_bounds))
            new_num_joints += 1
        return self._chain_factory(new_num_joints, new_joint_types, new_joint_axes, new_link_lengths)
    
    def crowding(self, population, fitness_scores):
        new_population = []
        while len(new_population) < self.population_size:
            idx1, idx2 = random.sample(range(len(population)), 2)
            child = self.mutate(self.crossover(population[idx1], population[idx2]))
            if not child.is_built:
                child.build_robot()
            child.load_robot()
            child.compute_chain_metrics(self.target_positions)
            # Precompute conditioning index in the main process
            child.conditioning_index = child.compute_global_conditioning_index()
            child_fitness = self.fitness(child)
            self.pyb.con.resetSimulation()
            self.pyb.enable_gravity()
            replace_idx = idx1 if (fitness_scores[idx1] > fitness_scores[idx2]) == self.sort_pop_low else idx2
            population[replace_idx], fitness_scores[replace_idx] = child, child_fitness
            new_population.append(child)
        return new_population
    
    def sort_population(self, population):
        """
        For each chain, update simulation metrics and precompute the conditioning index.
        Then, evaluate fitness in parallel using Ray.
        """
        self.population_pose_errors = []
        self.population_torques = []
        for chain in population:
            if not chain.is_built:
                chain.build_robot()
            chain.load_robot()
            chain.compute_chain_metrics(self.target_positions)
            chain.conditioning_index = chain.compute_global_conditioning_index()
            self.population_pose_errors.append(chain.mean_pose_error)
            self.population_torques.append(chain.mean_torque)
            self.pyb.con.resetSimulation()
            self.pyb.enable_gravity()
        fitness_scores = self.evaluate_population_fitness(population)
        combined = list(zip(population, fitness_scores))
        combined.sort(key=lambda x: x[1], reverse=not self.sort_pop_low)
        sorted_population, sorted_fitness = zip(*combined)
        return list(sorted_population), list(sorted_fitness)
    
    def run(self):
        population = [self.generate_random_chain() for _ in range(self.population_size)]
        total_chains_generated = self.population_size
        for gen in range(self.generations):
            population, fitness_scores = self.sort_population(population)
            print(f"Generation {gen + 1}: Best Fitness = {np.round(fitness_scores[0], 4)}")
            if gen == self.generations - 1:
                break
            population = self.crowding(population, fitness_scores)
        best_chain = population[0]
        best_chain.save_urdf('best_chain')
        return best_chain, total_chains_generated, self.generations


if __name__ == '__main__':
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
    target_poses = np.array(target_positions)
    save_urdf_dir = '/home/marcus/IMML/manipulator_codesign/manipulator_codesign/urdf/robots/'

    ga_pyb = GeneticAlgorithm(target_poses, save_urdf_dir=save_urdf_dir, backend='pybullet',
                              population_size=100, generations=20, renders=False, max_num_joints=7)
    
    best_chain, total_generated, total_iters = ga_pyb.run()
    print(f"\nTotal Kinematic Chains Generated: {total_generated}")
    print(f"Total Iterations: {total_iters}")
    best_chain.describe()
