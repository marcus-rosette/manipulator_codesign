import numpy as np
from scipy.spatial.transform import Rotation
import pickle
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.kinematic_chain import KinematicChainRTB, KinematicChainPyBullet


class KinematicChainProblem(Problem):
    def __init__(self, target_positions, backend='pybullet', renders=False, min_joints=2, max_joints=5, alpha=10.0, beta=0.1, delta=0.1, gamma=3.0):
        """
        Define the NSGA-II optimization problem for kinematic chain design.
        
        The decision vector has length:
            1 + (max_joints * 3)
        and four objectives are computed.
        """
        self.target_positions = np.array(target_positions)
        self.min_joints = min_joints
        self.max_joints = max_joints

        # Fitness function weights for each objective
        self.alpha = alpha  # Pose error penalty weight    
        self.beta = beta   # Joint torque penalty weight
        self.delta = delta  # Joint penalty weight
        self.gamma = gamma  # Conditioning index reward weight
        
        # Define lower and upper bounds for the decision vector.
        joint_type_bounds = [0, 1]  # Joint type: 0 (revolute), 1 (prismatic)
        joint_axis_bounds = [0, 2]  # Joint axis: 0 (x-axis), 1 (y-axis), 2 (z-axis)
        link_length_bounds = [0.1, 0.75]  # Link length: minimum 0.1, maximum 0.75

        # Lower and upper bounds for the decision vector.
        xl = [min_joints] + [joint_type_bounds[0], joint_axis_bounds[0], link_length_bounds[0]] * max_joints
        xu = [max_joints] + [joint_type_bounds[1], joint_axis_bounds[1], link_length_bounds[1]] * max_joints
        
        super().__init__(n_var=len(xl), n_obj=4, n_constr=0, xl=np.array(xl), xu=np.array(xu))
        
        self.backend = backend
        if self.backend == 'pybullet':
            # Initialize PyBullet utilities.
            from manipulator_codesign.pyb_utils import PybUtils
            from manipulator_codesign.load_objects import LoadObjects
            self.pyb = PybUtils(renders=renders)
            self.object_loader = LoadObjects(self.pyb.con)
        
    def _chain_factory(self, num_joints, joint_types, joint_axes, link_lengths):
        """
        Create a kinematic chain instance using the chosen backend.
        """
        if self.backend == 'pybullet':
            return KinematicChainPyBullet(self.pyb.con, num_joints, joint_types, joint_axes, link_lengths)
        else:
            return KinematicChainRTB(num_joints, joint_types, joint_axes, link_lengths)
        
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Batch evaluation of all individuals in the population.
        
        For each candidate, the chain is built, loaded, and its metrics are computed.
        The simulation is reset after each evaluation. Dynamic normalization factors are computed
        from the batch before scaling the four objectives.
        """
        exp_fit = False
        n_ind = X.shape[0]
        
        # Arrays to store raw metrics.
        raw_pose_errors = np.zeros(n_ind)
        raw_torques = np.zeros(n_ind)
        joint_counts = np.zeros(n_ind)
        conditioning_indices = np.zeros(n_ind)
        
        for i in range(n_ind):
            x = X[i, :]
            num_joints, joint_types, joint_axes, link_lengths = decode_decision_vector(x, self.min_joints, self.max_joints)
            
            # Create the chain.
            chain = self._chain_factory(num_joints, joint_types, joint_axes, link_lengths)
            
            # Build and load the robot if needed.
            if not chain.is_built:
                chain.build_robot()
            chain.load_robot()
            
            # Compute metrics (this should update chain.mean_pose_error, chain.mean_torque, etc.).
            chain.compute_chain_metrics(self.target_positions)
            
            raw_pose_errors[i] = chain.mean_pose_error
            raw_torques[i] = chain.mean_torque
            joint_counts[i] = chain.num_joints
            conditioning_indices[i] = chain.global_conditioning_index
            
            # Reset simulation to ensure a clean state for the next candidate.
            self.pyb.con.resetSimulation()
            self.pyb.enable_gravity()
        
        # Compute dynamic normalization factors from the current population.
        pose_error_norm = max(1e-6, np.max(raw_pose_errors))
        torque_norm = max(1e-6, np.max(raw_torques))
        
        # Compute the four objectives for each candidate.
        F = np.zeros((n_ind, 4))
        for i in range(n_ind):

            if exp_fit:
                normalized_pose_error = (raw_pose_errors[i] / pose_error_norm) ** 2
                normalized_torque_penalty = (raw_torques[i] / torque_norm) ** 2
                joint_penalty = joint_counts[i] / self.max_joints
                conditioning_index = conditioning_indices[i]
                
                # Scale the metrics using exponential decay or inverse functions.
                f1 = np.exp(-self.alpha * normalized_pose_error)      # Pose error objective (minimized)
                f2 = np.exp(-self.beta * normalized_torque_penalty)     # Torque penalty objective (minimized)
                f3 = np.exp(-self.delta * joint_penalty)                # Joint penalty objective (fewer joints are better)
                f4 = - (1 / (1 + abs(conditioning_index - 1)))     # Conditioning index objective (rewarding values closer to 1)

            else:
                # ** Linear Normalization of Penalties**
                normalized_pose_error = raw_pose_errors[i] / pose_error_norm
                normalized_torque_penalty = raw_torques[i] / torque_norm
                joint_penalty = joint_counts[i] / self.max_joints
                conditioning_index = conditioning_indices[i]

                target_pose_error = 0.0
                target_torque_error = 0.0
                target_joint_penalty = 0.0
                target_conditioning = 1.0

                f1 = self.alpha * abs(normalized_pose_error - target_pose_error)
                f2 = self.beta * abs(normalized_torque_penalty - target_torque_error)
                f3 = self.delta * abs(joint_penalty - target_joint_penalty)
                f4 = self.gamma * abs(conditioning_index - target_conditioning)

            F[i, 0] = f1
            F[i, 1] = f2
            F[i, 2] = f3
            F[i, 3] = f4
        
        out["F"] = F


if __name__ == '__main__':
    # Define target end-effector positions.
    target_positions = np.random.uniform(low=[-2.0, -2.0, 0], high=[2.0, 2.0, 2.0], size=(20, 3)).tolist()
    
    min_joints, max_joints = 3, 7
    problem = KinematicChainProblem(target_positions, min_joints=min_joints, max_joints=max_joints)
    
    # Configure the NSGA-II algorithm.
    algorithm = NSGA2(
        pop_size=500,  # Increased population size
        sampling=LatinHypercubeSampling(),  # More uniform initial sampling
        crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),  # Adjusted crossover parameters
        mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20),  # Adjusted mutation parameters
        n_offsprings=250,  # More offsprings per generation
        eliminate_duplicates=True
    )
    
    # Run the optimization.
    res = minimize(problem, algorithm, termination=('n_gen', 50), seed=1, verbose=True)

    # Define a dictionary to store relevant results
    results_dict = {
        "decision_vecs": res.X,  # Decision vectors
        "objective_vals": res.F,  # Objective values
        "min_joints": min_joints,
        "max_joints": max_joints,
        "pose_weight": problem.alpha,
        "torque_weight": problem.beta,
        "joint_weight": problem.delta,
        "conditioning_weight": problem.gamma,
    }

    storage_dir = '/home/marcus/IMML/manipulator_codesign/data/nsga2_results/'

    # Save results to a pickle file
    with open(f"{storage_dir}nsga2_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Results saved to {storage_dir}nsga2_results.pkl")
