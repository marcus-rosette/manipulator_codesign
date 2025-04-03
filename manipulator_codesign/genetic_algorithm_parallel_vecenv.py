import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import random
from manipulator_codesign.pyb_utils import PybUtils
from manipulator_codesign.load_objects import LoadObjects
from manipulator_codesign.kinematic_chain import KinematicChainRTB, KinematicChainPyBullet
import os
import warnings

# # Set environment variables to limit the number of CPU cores
# os.environ["OMP_NUM_THREADS"] = "4"  # Change this to the desired number of cores
# os.environ["MKL_NUM_THREADS"] = "4"  # This also affects some operations

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages

###########################################
# Top-Level Helper Functions (Picklable)  #
###########################################

def build_chain_from_params(params, backend, save_urdf_dir):
    """
    Reconstruct a kinematic chain from simple parameters.
    params: dict with keys 'num_joints', 'joint_types', 'joint_axes', 'link_lengths'
    """
    if backend == 'pybullet':
        # Create a new PybUtils instance inside the worker
        pyb = PybUtils(renders=False)
        return KinematicChainPyBullet(pyb.con, 
                                      params['num_joints'], 
                                      params['joint_types'], 
                                      params['joint_axes'], 
                                      params['link_lengths'], 
                                      save_urdf_dir=save_urdf_dir)
    else:
        return KinematicChainRTB(params['num_joints'], 
                                 params['joint_types'], 
                                 params['joint_axes'], 
                                 params['link_lengths'], 
                                 save_urdf_dir=save_urdf_dir)

def create_chain_fitness_env(chain_params_dict):
    """
    Create a ChainFitnessEnv instance from simple parameters.
    Expects chain_params_dict to have keys:
      'chain_params': dict with chain parameters,
      'target_positions': target positions,
      'backend': backend identifier,
      'save_urdf_dir': directory to save URDF,
      'max_num_joints': maximum joints,
      'renders': whether to render.
    """
    cp = chain_params_dict['chain_params']
    target_positions = chain_params_dict['target_positions']
    backend = chain_params_dict['backend']
    save_urdf_dir = chain_params_dict['save_urdf_dir']
    max_num_joints = chain_params_dict['max_num_joints']
    renders = chain_params_dict['renders']
    
    # Reconstruct the chain from parameters
    chain = build_chain_from_params(cp, backend, save_urdf_dir)
    return ChainFitnessEnv(chain, target_positions, backend, save_urdf_dir, max_num_joints, renders)

###########################################
# Environment Definition                  #
###########################################

class ChainFitnessEnv(gym.Env):
    def __init__(self, chain, target_positions, backend='pybullet', save_urdf_dir=None, max_num_joints=5, renders=False):
        super().__init__()
        self.target_positions = target_positions
        self.backend = backend
        self.save_urdf_dir = save_urdf_dir
        self.max_num_joints = max_num_joints
        self.chain = chain  # Reconstructed chain from parameters

        # Initialize PyBullet simulation (or other backend)
        self.pyb = PybUtils(renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)

        # Dummy spaces for evaluation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)

    def close(self):
        """Ensure proper cleanup of the PyBullet connection."""
        if self.backend == 'pybullet':
            self.pyb.disconnect()

    def reset(self):
        self.pyb.con.resetSimulation()
        self.pyb.enable_gravity()
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        """Evaluate the chain and return its fitness metrics."""
        if not self.chain.is_built:
            self.chain.build_robot()
        self.chain.load_robot()
        self.chain.compute_chain_metrics(self.target_positions)

        # Gather fitness metrics
        metrics = {
            "mean_pose_error": self.chain.mean_pose_error,
            "mean_torque": self.chain.mean_torque,
            "num_joints": self.chain.num_joints,
            "conditioning_index": self.chain.global_conditioning_index
        }

        new_obs = self.reset()
        return new_obs, 0.0, True, metrics  # done=True immediately

###########################################
# Genetic Algorithm Definition            #
###########################################

class GeneticAlgorithm:
    def __init__(self, target_positions, backend='pybullet', save_urdf_dir=None, max_num_joints=5, min_num_joints=3, 
                 link_len_bounds=(0.05, 1.0), sort_pop_low=True, population_size=20, generations=100, 
                 mutation_rate=0.3, crossover_rate=0.7, renders=False):
        self.target_positions = target_positions
        self.backend = backend
        self.save_urdf_dir = save_urdf_dir
        self.min_num_joints = min_num_joints
        self.max_num_joints = max_num_joints
        self.link_lengths_bounds = link_len_bounds
        self.sort_pop_low = sort_pop_low
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate  
        self.crossover_rate = crossover_rate
        self.renders = renders

        # For dynamic normalization
        self.population_pose_errors = []
        self.population_torques = []

        if self.backend == 'pybullet':
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
        num_joints = random.randint(self.min_num_joints, self.max_num_joints)
        joint_types = [random.choice([0, 1]) for _ in range(num_joints)]
        joint_axes = [random.choice(['x', 'y', 'z']) for _ in range(num_joints)]
        link_lengths = [random.uniform(*self.link_lengths_bounds) for _ in range(num_joints)]
        return self._chain_factory(num_joints, joint_types, joint_axes, link_lengths)
    
    def fitness(self, chain):
        alpha = 10
        beta = 0.1
        delta = 0.1
        gamma = 3.0

        pose_error_norm_default = 100
        torque_norm_default = 100
        pose_error_norm = np.max(self.population_pose_errors) if len(self.population_pose_errors) > 0 else pose_error_norm_default
        torque_norm = np.max(self.population_torques) if len(self.population_torques) > 0 else torque_norm_default

        normalized_pose_error = (chain.mean_pose_error / pose_error_norm)
        normalized_torque_penalty = (chain.mean_torque / torque_norm)
        joint_penalty = chain.num_joints / self.max_num_joints
        conditioning_index = chain.global_conditioning_index

        fitness_score = (
            alpha * normalized_pose_error +
            beta * normalized_torque_penalty +
            delta * joint_penalty -
            gamma * conditioning_index
        )
        return fitness_score
    
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
                delta = random.gauss(0, 0.1)
                new_link_lengths[i] = np.clip(new_link_lengths[i] + delta, *self.link_lengths_bounds)
        
        if random.random() < self.mutation_rate:
            idx = random.randint(0, new_num_joints - 1)
            new_joint_types[idx] = 1 - new_joint_types[idx]
        
        if random.random() < self.mutation_rate:
            idx = random.randint(0, new_num_joints - 1)
            new_joint_axes[idx] = random.choice(['x', 'y', 'z'])
        
        if random.random() < self.mutation_rate and new_num_joints > self.min_num_joints:
            idx = random.randint(0, new_num_joints - 1)
            del new_joint_types[idx]
            del new_joint_axes[idx]
            del new_link_lengths[idx]
            new_num_joints -= 1

        if random.random() < self.mutation_rate and new_num_joints < self.max_num_joints:
            new_joint_types.append(random.choice([0, 1]))
            new_joint_axes.append(random.choice(['x', 'y', 'z']))
            new_link_lengths.append(random.uniform(*self.link_lengths_bounds))
            new_num_joints += 1

        return self._chain_factory(new_num_joints, new_joint_types, new_joint_axes, new_link_lengths)
    
    def chain_distance(self, chain1, chain2, w_joint_type=1.0, w_joint_axis=1.0, w_link_length=1.0, w_joint_count=2.0):
        joint_count_diff = abs(chain1.num_joints - chain2.num_joints)
        distance = w_joint_count * joint_count_diff
        min_joints = min(chain1.num_joints, chain2.num_joints)
        for i in range(min_joints):
            type_diff = 0 if chain1.joint_types[i] == chain2.joint_types[i] else 1
            axis_diff = 0 if chain1.joint_axes[i] == chain2.joint_axes[i] else 1
            length_diff = abs(chain1.link_lengths[i] - chain2.link_lengths[i])
            distance += (w_joint_type * type_diff + w_joint_axis * axis_diff + w_link_length * length_diff)
        return distance

    def crowding(self, population, fitness_scores):
        new_population = []
        while len(new_population) < self.population_size:
            idx1, idx2 = random.sample(range(len(population)), 2)
            parent1, parent2 = population[idx1], population[idx2]
            child = self.mutate(self.crossover(parent1, parent2))
            if not child.is_built:
                child.build_robot()
            child.load_robot()
            child.compute_chain_metrics(self.target_positions)
            child_fitness = self.fitness(child)
            dist1 = self.chain_distance(child, parent1)
            dist2 = self.chain_distance(child, parent2)
            if dist1 < dist2:
                if child_fitness < fitness_scores[idx1]:
                    population[idx1], fitness_scores[idx1] = child, child_fitness
            else:
                if child_fitness < fitness_scores[idx2]:
                    population[idx2], fitness_scores[idx2] = child, child_fitness
            new_population.append(child)
            self.pyb.con.resetSimulation()
            self.pyb.enable_gravity()
        return new_population
    
    def evaluate_population_parallel(self, population):
        num_candidates = len(population)
        # For each chain, extract only the parameters needed to rebuild it
        chain_params_list = []
        for chain in population:
            params = {
                'num_joints': chain.num_joints,
                'joint_types': chain.joint_types,
                'joint_axes': chain.joint_axes,
                'link_lengths': chain.link_lengths
            }
            chain_params_list.append(params)
        
        # Build list of picklable dictionaries for each environment
        env_param_list = [{
            'chain_params': cp,
            'target_positions': self.target_positions,
            'backend': self.backend,
            'save_urdf_dir': self.save_urdf_dir,
            'max_num_joints': self.max_num_joints,
            'renders': self.renders
        } for cp in chain_params_list]

        envs = DummyVecEnv([lambda params=ep: create_chain_fitness_env(params) for ep in env_param_list])
        envs.reset()
        actions = [0] * num_candidates  # One dummy action per environment
        _, _, dones, infos = envs.step(actions)
        
        fitness_scores = []
        all_metrics = []
        for i, done in enumerate(dones):
            if done:
                metrics = infos[i]
                if "mean_pose_error" in metrics and "mean_torque" in metrics:
                    fitness = (
                        10 * metrics["mean_pose_error"] +
                        0.1 * metrics["mean_torque"] +
                        0.1 * metrics["num_joints"] -
                        3.0 * metrics["conditioning_index"]
                    )
                    fitness_scores.append(fitness)
                    all_metrics.append(metrics)
                else:
                    print(f"Incomplete metrics for individual {i}: {metrics}")
        envs.close()
        return fitness_scores, all_metrics

    def sort_population_parallel(self, population):
        fitness_scores, metrics_list = self.evaluate_population_parallel(population)
        self.population_pose_errors = [m["mean_pose_error"] for m in metrics_list]
        self.population_torques = [m["mean_torque"] for m in metrics_list]
        # print(f"Population size: {len(population)}, Fitness scores size: {len(fitness_scores)}")
        if len(fitness_scores) != len(population):
            raise ValueError("Mismatch: fitness_scores and population size don't match!")
        return population, fitness_scores

    def run(self):
        population = [self.generate_random_chain() for _ in range(self.population_size)]
        total_chains_generated = self.population_size
        
        for gen in range(self.generations + 1):
            self.population_pose_errors = []
            self.population_torques = []
            population, fitness_scores = self.sort_population_parallel(population)
            print(f"Generation {gen + 1}: Best Fitness = {np.round(fitness_scores[0], 4)}")
            if gen == self.generations:
                break
            population = self.crowding(population, fitness_scores)
            total_chains_generated += len(population)
        
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
    save_urdf_dir = '/home/marcus/IMML/manipulator_codesign/manipulator_codesign/urdf/robots/'
    ga_pyb = GeneticAlgorithm(target_positions, save_urdf_dir=save_urdf_dir, backend='pybullet', 
                              population_size=50, generations=10, renders=False, max_num_joints=7)
    
    best_chain, total_generated, total_iters = ga_pyb.run()
    print(f"\nTotal Kinematic Chains Generated: {total_generated}")
    print(f"Total Iterations: {total_iters}")
    best_chain.describe()
