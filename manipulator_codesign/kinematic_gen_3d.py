import numpy as np
import random
from urdf_gen import URDFGen
from pyb_utils import PybUtils
from load_objects import LoadObjects
from load_robot import LoadRobot


class KinematicChain:
    def __init__(self, pyb_con, num_joints, joint_types, joint_axes, link_lengths):
        self.pyb_con = pyb_con
        self.num_joints = num_joints
        self.joint_types = joint_types
        self.joint_axes = joint_axes
        self.link_lengths = link_lengths

        self.joint_limit_prismatic = (-0.5, 0.5)
        self.joint_limit_revolute = (-np.pi, np.pi)

        # Map joint limits based on joint type
        self.joint_limits = [self.joint_limit_prismatic if jt == 0 else self.joint_limit_revolute for jt in joint_types]

        robot_name = 'silly_robot'
        save_urdf_dir = '/home/marcus/IMML/manipulator_codesign/manipulator_codesign/gen_urdf_files/'

        self.urdf_gen = URDFGen(robot_name, save_urdf_dir)
        self.urdf_gen.create_manipulator(joint_axes, joint_types, link_lengths, self.joint_limits)
        urdf_path = self.urdf_gen.save_temp_urdf()

        self.robot = LoadRobot(self.pyb_con, 
                          urdf_path, 
                          start_pos=[0, 0, 0], 
                          start_orientation=self.pyb_con.getQuaternionFromEuler([0, 0, 0]),
                          home_config=[0] * num_joints)

    def save_urdf(self, filename):
        """Save the URDF representation of this kinematic chain."""
        self.urdf_gen.save_urdf(filename)

    def describe(self):
        """Prints the details of the kinematic chain."""
        print(f"\nKinematic Chain Description:")
        print(f"Number of Joints: {self.num_joints}")
        print(f"Joint Types: {self.joint_types}")
        print(f"Joint Axes: {self.joint_axes}")
        print(f"Link Lengths: {self.link_lengths}")


class GeneticAlgorithm:
    def __init__(self, target_positions, population_size=20, generations=100, mutation_rate=0.3, crossover_rate=0.7, renders=False):
        # Genetic Algorithm Parameters
        self.target_positions = np.array(target_positions)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate  
        self.crossover_rate = crossover_rate

        self.link_lengths_bounds = (0.1, 0.75)

        # Pybullet Setup
        self.pyb = PybUtils(self, renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)
    
    def generate_random_chain(self, max_joints=5):
        """Creates a random kinematic chain and returns an instance of KinematicChain."""
        num_joints = random.randint(2, max_joints)
        joint_types = [random.choice([0, 1]) for _ in range(num_joints)]  # 0 for prismatic, 1 for revolute
        joint_axes = [random.choice(['x', 'y', 'z']) for _ in range(num_joints)]  # Random axis for each joint
        link_lengths = [random.uniform(self.link_lengths_bounds[0], self.link_lengths_bounds[1]) for _ in range(num_joints)]  # Randomized link lengths
        return KinematicChain(self.pyb.con, num_joints, joint_types, joint_axes, link_lengths)
    
    def fitness(self, chain):
        """Evaluate fitness across multiple target positions by computing average end-effector error."""
        total_error = 0.0

        for target in self.target_positions:
            joint_config = chain.robot.inverse_kinematics(target, pos_tol=0.1)
            chain.robot.reset_joint_positions(joint_config)
            ee_pos, _ = chain.robot.get_link_state(chain.robot.end_effector_index)
            
            # Compute Euclidean distance error for this target
            total_error += np.linalg.norm(target - ee_pos)

        # Return the average error across all targets
        return total_error / len(self.target_positions)
    
    def crossover(self, parent1, parent2):
        """Create a new KinematicChain by mixing genes from two parents."""
        # Choose the child's number of joints:
        if parent1.num_joints == parent2.num_joints:
            child_num_joints = parent1.num_joints
        else:
            # You could choose one parent's joint count randomly:
            child_num_joints = random.choice([parent1.num_joints, parent2.num_joints])
        
        child_joint_types = []
        child_joint_axes = []
        child_link_lengths = []
        
        for i in range(child_num_joints):
            # Check if both parents have a joint at index i
            gene_from_parent1 = i < parent1.num_joints
            gene_from_parent2 = i < parent2.num_joints
            
            if gene_from_parent1 and gene_from_parent2:
                # For joint type and axis (discrete), choose randomly
                joint_type = random.choice([parent1.joint_types[i], parent2.joint_types[i]])
                joint_axis = random.choice([parent1.joint_axes[i], parent2.joint_axes[i]])
                # For link length (continuous), take an average (or weighted average)
                link_length = (parent1.link_lengths[i] + parent2.link_lengths[i]) / 2
            elif gene_from_parent1:
                joint_type = parent1.joint_types[i]
                joint_axis = parent1.joint_axes[i]
                link_length = parent1.link_lengths[i]
            elif gene_from_parent2:
                joint_type = parent2.joint_types[i]
                joint_axis = parent2.joint_axes[i]
                link_length = parent2.link_lengths[i]
            
            child_joint_types.append(joint_type)
            child_joint_axes.append(joint_axis)
            child_link_lengths.append(link_length)
        
        # Create and return a new KinematicChain instance
        # Note: We use parent1's pyb_con (they should both have the same connection).
        child_chain = KinematicChain(parent1.pyb_con, child_num_joints, child_joint_types, child_joint_axes, child_link_lengths)
        return child_chain

    def mutate(self, chain):
        """Incrementally mutate a chain's parameters rather than generating an entirely new chain."""
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
                new_link_lengths[i] = np.clip(new_link_lengths[i] + delta, self.link_lengths_bounds[0], self.link_lengths_bounds[1])
        
        # Mutate joint types with a low probability (flip from prismatic (0) to revolute (1) or vice versa)
        if random.random() < (self.mutation_rate / 2):
            idx = random.randint(0, new_num_joints - 1)
            new_joint_types[idx] = 1 - new_joint_types[idx]
        
        # Mutate joint axes with a low probability (pick a new axis)
        if random.random() < (self.mutation_rate / 2):
            idx = random.randint(0, new_num_joints - 1)
            new_joint_axes[idx] = random.choice(['x', 'y', 'z'])
        
        # Occasionally, adjust the joint count: remove a joint (if more than 2) or add one (if below max, e.g., 5)
        if random.random() < (self.mutation_rate / 4):
            if new_num_joints > 2:
                idx = random.randint(0, new_num_joints - 1)
                del new_joint_types[idx]
                del new_joint_axes[idx]
                del new_link_lengths[idx]
                new_num_joints -= 1
        if random.random() < (self.mutation_rate / 4):
            if new_num_joints < 5:
                new_joint_types.append(random.choice([0, 1]))
                new_joint_axes.append(random.choice(['x', 'y', 'z']))
                new_link_lengths.append(random.uniform(self.link_lengths_bounds[0], self.link_lengths_bounds[1]))
                new_num_joints += 1

        # Create a new mutated chain using the modified parameters
        return KinematicChain(chain.pyb_con, new_num_joints, new_joint_types, new_joint_axes, new_link_lengths)
    
    def run(self, error_threshold=0.005):
        """Run the genetic algorithm optimization for kinematic chain design."""
        population = [self.generate_random_chain() for _ in range(self.population_size)]
        total_chains_generated = self.population_size
        total_iterations = 0
        
        for gen in range(self.generations):
            total_iterations += 1
            
            # Sort population based on fitness (lower error is better)
            population = sorted(population, key=lambda x: self.fitness(x), reverse=False)
            best_error = self.fitness(population[0])
        
            # Print progress for this generation
            print(f"Generation {gen + 1}: Best Error = {np.round(best_error, 4)}")
            
            # Stop if best chain achieves near-zero error across all targets
            if best_error < error_threshold:
                print(f"Converged in {gen} generations. Stopping early.")
                break
            
            # Keep the top 5 chains and generate new ones
            new_population = population[:5]
            
            # Generate new offspring until the population is replenished
            while len(new_population) < self.population_size:
                # Select two parents from the top-performing individuals
                parent1, parent2 = random.sample(population[:10], 2)

                # Use the crossover rate to decide whether to crossover or simply clone a parent
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1

                # Mutate the child further
                child = self.mutate(child)
                new_population.append(child)
                total_chains_generated += 1
            
            population = new_population
        
        best_chain = population[0]  # Best kinematic chain

        # Save the best URDF
        best_chain.save_urdf('best_chain')
        
        return best_chain, total_chains_generated, total_iterations


if __name__ == '__main__':
    # Example Target End-Effector Position
    target_positions = [
        [1.0, 2.0, 1.0],
        [0.5, 1.5, 1.2],
        [-0.5, 1.0, 0.8],
        [0.0, 1.0, 1.0],
        [-1.0, 1.5, 1.2],
        [-0.5, 0.5, 0.1],
        [0.5, 0.5, 0.1],
    ]

    # Instantiate and run the Genetic Algorithm
    ga = GeneticAlgorithm(target_positions, population_size=40)
    best_chain, total_chains_generated, total_iterations = ga.run(error_threshold=0.009)

    # Print details of the best kinematic chain
    print("\nBest Kinematic Chain Found:")
    best_chain.describe()

    print(f"\nTotal Kinematic Chains Generated: {total_chains_generated}")
    print(f"Total Iterations: {total_iterations}")
