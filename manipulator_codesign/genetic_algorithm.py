import numpy as np
import random
from scipy.spatial.transform import Rotation
from manipulator_codesign.kinematic_chain import KinematicChainRTB, KinematicChainPyBullet


class GeneticAlgorithm:
    def __init__(self, target_positions, backend='pybullet', save_urdf_dir=None, max_num_joints=5, link_len_bounds=(0.05, 1.0),
                 population_size=20, generations=100, mutation_rate=0.3, crossover_rate=0.7, joint_penalty_weight=0.1, renders=False):
        """
        Initialize the genetic algorithm for manipulator codesign.
        Args:
            target_positions (list or array-like): The target positions for the manipulator to reach.
            backend (str, optional): The backend to use for simulation ('rtb' or 'pybullet'). Default is 'rtb'.
            save_urdf_dir (str, optional): Directory to save URDF files. Default is None.
            population_size (int, optional): The size of the population for the genetic algorithm. Default is 20.
            generations (int, optional): The number of generations to run the genetic algorithm. Default is 100.
            mutation_rate (float, optional): The mutation rate for the genetic algorithm. Default is 0.3.
            crossover_rate (float, optional): The crossover rate for the genetic algorithm. Default is 0.7.
            renders (bool, optional): Whether to render the simulation (only applicable for 'pybullet' backend). Default is False.
        """
        self.target_positions = target_positions
        self.backend = backend
        self.save_urdf_dir = save_urdf_dir
        self.max_num_joints = max_num_joints
        self.link_lengths_bounds = link_len_bounds
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate  
        self.crossover_rate = crossover_rate
        self.joint_penalty_weight = joint_penalty_weight

        # Population statistics for dynamic normalization
        self.population_pose_errors = []  # To track pose errors across generations
        self.population_torques = []  # To track torque magnitudes across generations

        if self.backend == 'pybullet':
            # For PyBullet, initialize your connection and any other required objects.
            from manipulator_codesign.pyb_utils import PybUtils
            from manipulator_codesign.load_objects import LoadObjects
            self.pyb = PybUtils(renders=renders)
            self.object_loader = LoadObjects(self.pyb.con)
    
    def _chain_factory(self, num_joints, joint_types, joint_axes, link_lengths):
        """
        Creates and returns a kinematic chain object based on the specified backend.
        Args:
            num_joints (int): The number of joints in the kinematic chain.
            joint_types (list): A list specifying the types of each joint.
            joint_axes (list): A list specifying the axes of each joint.
            link_lengths (list): A list specifying the lengths of each link.
        Returns:
            KinematicChainPyBullet or KinematicChainRTB: An instance of the kinematic chain class 
            based on the specified backend.
        """

        if self.backend == 'pybullet':
            return KinematicChainPyBullet(self.pyb.con, num_joints, joint_types, joint_axes, link_lengths, save_urdf_dir=self.save_urdf_dir)
        else:  # default to RTB
            return KinematicChainRTB(num_joints, joint_types, joint_axes, link_lengths, save_urdf_dir=self.save_urdf_dir)
    
    def generate_random_chain(self):
        """
        Generates a random kinematic chain for a manipulator.
        This function creates a random kinematic chain with a specified maximum number of joints.
        Each joint can be of a random type (e.g., revolute or prismatic) and oriented along a random axis.
        The lengths of the links between the joints are also randomly generated within specified bounds.
        Returns:
            KinematicChain: A kinematic chain object created by the _chain_factory method.
        """

        num_joints = random.randint(2, self.max_num_joints)
        joint_types = [random.choice([0, 1]) for _ in range(num_joints)]
        joint_axes = [random.choice(['x', 'y', 'z']) for _ in range(num_joints)]
        link_lengths = [random.uniform(*self.link_lengths_bounds) for _ in range(num_joints)]
        return self._chain_factory(num_joints, joint_types, joint_axes, link_lengths)
    
    def fitness(self, chain):
        """
        Calculate the fitness of a given chain based on target positions.

        The fitness is computed as the average error between the chain's computed
        positions and the target positions.

        Args:
            chain (KinematicChain): The chain object whose fitness is to be calculated.

        Returns:
            float: The average error representing the fitness of the chain.
        """
        # Weights for each term
        alpha = 0.1  # Pose error penalty weight    
        beta = 0.001   # Joint torque penalty weight
        delta = 0.001  # Joint penalty weight
        gamma = 3.0  # Conditioning index reward weight

        # Dynamic normalization factors based on population trends
        pose_error_norm = max(1e-6, np.max(self.population_pose_errors))  
        torque_norm = max(1e-6, np.max(self.population_torques))

        # ** Quadratic Normalization of Penalties**
        normalized_pose_error = (chain.mean_pose_error / pose_error_norm) ** 2
        normalized_torque_penalty = (chain.mean_torque / torque_norm) ** 2
        joint_penalty = chain.num_joints / self.max_num_joints

        # ** Reward global conditioning numbers that are closer to 1 (Using an exponential function to penalize deviation from 1)
        conditioning_index = chain.compute_global_conditioning_index()

        # Scaled metrics
        normalized_pose_error_scaled = np.exp(-alpha * normalized_pose_error)  # Exponential decay for pose error
        normalized_torque_penalty_scaled = np.exp(-beta * normalized_torque_penalty)  # Exponential decay for torque penalty
        joint_penalty_scaled = np.exp(-delta * joint_penalty)  # Exponential decay for joint penalty
        # conditioning_index_scaled = np.exp(-gamma * abs(conditioning_index - 1)) # Use an exponential decay to penalize deviation from 1
        conditioning_index_scaled = 1 / (1 + abs(conditioning_index - 1)) # Use an inverse function to penalize deviation from 1

        # print(f"Pose Error: {np.round(normalized_pose_error_scaled, 4)}, Joint Penalty: {np.round(joint_penalty_scaled, 4)}, "
        #   f"Torque Penalty: {np.round(normalized_torque_penalty_scaled, 4)}, Conditioning Index: {np.round(conditioning_index_scaled, 4)}")
    
        fitness_score = (
            normalized_pose_error_scaled *
            normalized_torque_penalty_scaled *
            joint_penalty_scaled *
            conditioning_index_scaled
        )
        
        return fitness_score
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent chains to produce a child chain.
        This method combines the genetic information from two parent chains to create a new child chain.
        The number of joints in the child chain is determined by randomly selecting the number of joints
        from either parent if they differ. For each joint, the type and axis are randomly chosen from
        either parent, and the link length is averaged if both parents have the joint at that position.
        Args:
            parent1 (KinematicChain): The first parent kinematic chain.
            parent2 (KinematicChain): The second parent kinematic chain.
        Returns:
            KinematicChain: A new kinematic chain created by combining the genetic information from the two parents.
        """
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
            else:  # gene_from_parent2 must be true
                joint_type = parent2.joint_types[i]
                joint_axis = parent2.joint_axes[i]
                link_length = parent2.link_lengths[i]
            
            child_joint_types.append(joint_type)
            child_joint_axes.append(joint_axis)
            child_link_lengths.append(link_length)

        return self._chain_factory(child_num_joints, child_joint_types, child_joint_axes, child_link_lengths)
    
    def mutate(self, chain):
        """
        Incrementally mutate a kinematic chain's parameters rather than generating an entirely new chain.

        Args:
            chain (KinematicChain): The kinematic chain to be mutated.

        Returns:
            KinematicChain: A new kinematic chain with mutated parameters.
        """
        # Copy the current chain's parameters
        new_num_joints = chain.num_joints
        new_joint_types = chain.joint_types.copy()
        new_joint_axes = chain.joint_axes.copy()
        new_link_lengths = chain.link_lengths.copy()
        
        # Mutate each link length by adding a small perturbation
        for i in range(new_num_joints):
            if random.random() < self.mutation_rate:
                # Apply a small Gaussian perturbation; adjust the sigma value as needed
                delta = random.gauss(0, 0.05)
                # Clip the new length to the allowed bounds
                new_link_lengths[i] = np.clip(new_link_lengths[i] + delta, *self.link_lengths_bounds)
        
        # Mutate joint types with a low probability (flip from prismatic (0) to revolute (1) or vice versa)
        if random.random() < (self.mutation_rate / 2):
            idx = random.randint(0, new_num_joints - 1)
            new_joint_types[idx] = 1 - new_joint_types[idx]
        
        # Mutate joint axes with a low probability (pick a new axis)
        if random.random() < (self.mutation_rate / 2):
            idx = random.randint(0, new_num_joints - 1)
            new_joint_axes[idx] = random.choice(['x', 'y', 'z'])
        
        # Occasionally, adjust the joint count: remove a joint (if more than 2) or add one (if below max)
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
    
    def sort_population(self, population):
        """
        Score the population of kinematic chains based on their fitness scores.
        
        Args:
            population (list): A list of kinematic chain objects to be scored.
        
        Returns:
            list: A list of fitness scores corresponding to each kinematic chain in the population.
        """
        # Compute fitness for each chain and store the pose errors and torques
        chain_fitness = []
        for chain in population:
            if not chain.is_built:
                chain.build_robot()

            chain.load_robot()  # Load the robot into the simulation

            chain.compute_chain_metrics(self.target_positions)
            self.population_pose_errors.append(chain.mean_pose_error)
            self.population_torques.append(chain.mean_torque)

            fitness_score = self.fitness(chain)
            chain_fitness.append((chain, fitness_score))

            self.pyb.con.resetSimulation()
            self.pyb.enable_gravity()

        # Sort population based on fitness (higher fitness is better)
        chain_fitness.sort(key=lambda x: x[1], reverse=True)
        population = [fs[0] for fs in chain_fitness]
        fitness_scores = [fs[1] for fs in chain_fitness]
        return population, fitness_scores
    
    def run(self):
        """
        Run the genetic algorithm optimization for kinematic chain design.

        Returns:
            tuple: A tuple containing:
            - best_chain (KinematicChain): The best kinematic chain found by the algorithm.
            - total_chains_generated (int): The total number of chains generated during the run.
            - total_iterations (int): The total number of generations processed.
        """
        # Generate and initialize the initial population of kinematic chains
        population = [self.generate_random_chain() for _ in range(self.population_size)]
        total_chains_generated = self.population_size
        
        for gen in range(self.generations + 1):
            # At the beginning of each generation
            self.population_pose_errors = []
            self.population_torques = []

            # Sort population based on fitness (higher error is better)
            population, fitness_scores = self.sort_population(population)
        
            # Print progress for this generation
            print(f"Generation {gen + 1}: Best Fitness = {np.round(fitness_scores[0], 4)}")

            if gen == self.generations:
                break
            
            # Keep the top 30% chains and generate new ones
            top_percent = max(1, int(self.population_size * 0.3))
            top_population = population[:top_percent]
            new_population = top_population.copy()

            # Determine how many offspring you need
            num_offspring = self.population_size - len(top_population)

            # Precompute parent indices and crossover decisions
            parent_indices = np.random.randint(0, len(top_population), size=(num_offspring, 2))
            do_crossover = np.random.rand(num_offspring) < self.crossover_rate

            # Generate new offspring
            for i, (idx1, idx2) in enumerate(parent_indices):
                parent1 = top_population[idx1]
                parent2 = top_population[idx2]
                if do_crossover[i]:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1  # Clone parent1 if no crossover
                child = self.mutate(child)
                new_population.append(child)
                total_chains_generated += 1
            
            population = new_population

        best_chain = population[0]
        best_chain.save_urdf('best_chain')
        
        return best_chain, total_chains_generated, self.generations


if __name__ == '__main__':
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

    target_poses = np.array(target_positions)

    save_urdf_dir = '/home/marcus/IMML/manipulator_codesign/manipulator_codesign/urdf/robots/'

    ga_pyb = GeneticAlgorithm(target_poses, save_urdf_dir=save_urdf_dir, backend='pybullet', 
                              population_size=10, generations=10, renders=False, max_num_joints=7)
    
    best_chain, total_generated, total_iters = ga_pyb.run()

    print(f"\nTotal Kinematic Chains Generated: {total_generated}")
    print(f"Total Iterations: {total_iters}")
    best_chain.describe()
