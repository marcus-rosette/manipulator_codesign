import numpy as np
from scipy.spatial.transform import Rotation
import pickle
import time
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

        # Number of objectives
        self.n_obj = 6 
        
        # Define lower and upper bounds for the decision vector.
        joint_type_bounds = [0, 1]  # Joint type: 0 (revolute), 1 (prismatic)
        joint_axis_bounds = [0, 2]  # Joint axis: 0 (x-axis), 1 (y-axis), 2 (z-axis)
        link_length_bounds = [0.1, 0.75]  # Link length: minimum 0.1, maximum 0.75

        # Lower and upper bounds for the decision vector.
        xl = [min_joints] + [joint_type_bounds[0], joint_axis_bounds[0], link_length_bounds[0]] * max_joints
        xu = [max_joints] + [joint_type_bounds[1], joint_axis_bounds[1], link_length_bounds[1]] * max_joints
        
        super().__init__(n_var=len(xl), n_obj=self.n_obj, n_constr=0, xl=np.array(xl), xu=np.array(xu))
        
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
        manip_scores_rrmc = np.zeros(n_ind)
        delta_joint_score_rrmc = np.zeros(n_ind)
        raw_pos_errors_rrmc = np.zeros(n_ind)
        raw_ori_errors_rrmc = np.zeros(n_ind)
        
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
            manip_scores_rrmc[i] = chain.mean_manip_score_rrmc
            delta_joint_score_rrmc[i] = chain.mean_delta_joint_score_rrmc
            raw_pos_errors_rrmc[i] = chain.mean_pos_error_rrmc
            raw_ori_errors_rrmc[i] = chain.mean_ori_error_rrmc
            
            # Reset simulation to ensure a clean state for the next candidate.
            self.pyb.con.resetSimulation()
            self.pyb.enable_gravity()
        
        # Compute dynamic normalization factors from the current population. ------------------- NOT WORKING TOO WELL
        # pose_error_norm = max(1e-6, np.max(raw_pose_errors))
        # torque_norm = max(1e-6, np.max(raw_torques))
        # manip_score_rrmc_norm = max(1e-6, np.max(manip_scores_rrmc))
        # delta_joint_score_rrmc_norm = max(1e-6, np.max(delta_joint_score_rrmc))
        # pos_error_norm_rrmc = max(1e-6, np.max(raw_pos_errors_rrmc))
        # ori_error_norm_rrmc = max(1e-6, np.max(raw_ori_errors_rrmc))

        # Compute robust scale factors for each metric using IQR.
        pose_error_norm = self.robust_log_normalize(raw_pose_errors)
        torque_norm = self.robust_log_normalize(raw_torques)
        manip_score_rrmc_norm = self.robust_log_normalize(manip_scores_rrmc)
        delta_joint_score_rrmc_norm = self.robust_log_normalize(delta_joint_score_rrmc)
        pos_error_norm_rrmc = self.robust_log_normalize(raw_pos_errors_rrmc)
        ori_error_norm_rrmc = self.robust_log_normalize(raw_ori_errors_rrmc)
        
        # Compute the objectives for each candidate.
        F = np.zeros((n_ind, self.n_obj))
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
                # normalized_pose_error = raw_pose_errors[i] / pose_error_norm
                # normalized_torque_penalty = raw_torques[i] / torque_norm
                joint_penalty = joint_counts[i] / self.max_joints
                conditioning_index = conditioning_indices[i]
                # normalized_manip_score_rrmc = manip_scores_rrmc[i] / manip_score_rrmc_norm
                # normalized_delta_joint_score_rrmc = delta_joint_score_rrmc[i] / delta_joint_score_rrmc_norm
                # normalized_pos_error_rrmc = raw_pos_errors_rrmc[i] / pos_error_norm_rrmc
                # normalized_ori_error_rrmc = raw_ori_errors_rrmc[i] / ori_error_norm_rrmc
                normalized_pose_error = pose_error_norm[i]
                normalized_torque_penalty = torque_norm[i]
                normalized_manip_score_rrmc = manip_score_rrmc_norm[i]
                normalized_delta_joint_score_rrmc = delta_joint_score_rrmc_norm[i]
                normalized_pos_error_rrmc = pos_error_norm_rrmc[i]
                normalized_ori_error_rrmc = ori_error_norm_rrmc[i]

                target_pose_error = 0.0
                target_torque_error = 0.0
                target_joint_penalty = 0.0
                target_conditioning = 1.0

                f1 = self.alpha * abs(normalized_pose_error - target_pose_error)
                f2 = self.beta * abs(normalized_torque_penalty - target_torque_error)
                f3 = self.delta * abs(joint_penalty - target_joint_penalty)
                f4 = self.gamma * abs(conditioning_index - target_conditioning)

                # TODO: Define target values for resolved rate motion control (RRMC).
                target_manip_score_rrmc = 4.0
                target_delta_joint_score_rrmc = 0.0
                target_pos_error_rrmc = 0.0
                target_ori_error_rrmc = 0.0

                manip_rrmc_scalar = 1.0
                delta_joint_score_rrmc_scalar = 1.0
                pos_error_rrmc_scalar = 1.0
                ori_error_rrmc_scalar = 1.0

                # f5 = manip_rrmc_scalar * abs(normalized_manip_score_rrmc - target_manip_score_rrmc)
                # f6 = delta_joint_score_rrmc_scalar * abs(normalized_delta_joint_score_rrmc - target_delta_joint_score_rrmc)
                # f7 = pos_error_rrmc_scalar * abs(normalized_pos_error_rrmc - target_pos_error_rrmc)
                # f8 = ori_error_rrmc_scalar * abs(normalized_ori_error_rrmc - target_ori_error_rrmc)
                f5 = normalized_manip_score_rrmc
                f6 = normalized_delta_joint_score_rrmc
                f7 = normalized_pos_error_rrmc
                f8 = normalized_ori_error_rrmc

            F[i, 0] = f1
            F[i, 1] = f2
            F[i, 2] = f3
            F[i, 3] = f4

            # Delta joint error
            F[i, 4] = f6

            # Position error 
            F[i, 5] = f7

            # F[i, 4] = f5
            # F[i, 5] = f6
            # F[i, 6] = f7
            # F[i, 7] = f8
        
        out["F"] = F

    @staticmethod
    def robust_log_normalize(data):
        """
        Apply a logarithmic transformation to compress the scale and then perform min-max normalization.
        
        Parameters:
            data (array-like): Input array of penalty values.
        
        Returns:
            np.array: Normalized values between 0 and 1.
        """
        # Apply log1p to avoid issues with zero values
        log_data = np.log1p(data)
        
        # Min-max normalization on the log-transformed data
        min_val = np.min(log_data)
        max_val = np.max(log_data)
        
        # Avoid division by zero in case max == min
        if max_val - min_val < 1e-6:
            return np.zeros_like(log_data)
        
        normalized = (log_data - min_val) / (max_val - min_val)
        return normalized
    
    @staticmethod
    def tanh_normalize_to_01(data):
        mean = np.mean(data)
        std = np.std(data)
        
        # Avoid division by zero
        std = std if std != 0 else 1e-8

        normalized = (1 + np.tanh((data - mean) / std)) / 2
        return normalized

class TimeTrackingCallback:
    def __init__(self, total_generations):
        self.total_generations = total_generations
        self.start_time = None

    def __call__(self, algorithm):
        if algorithm.n_gen == 1:
            self.start_time = time.time()
            print("=" * 45)
            print(f"| Gen | Elapsed Time | Est. Remaining Time |")
            print("=" * 45)
            print(f'| 1/{self.total_generations} | --- | --- |')
        elif self.start_time is not None:
            elapsed = time.time() - self.start_time
            avg_time_per_gen = elapsed / (algorithm.n_gen - 1)
            remaining_gens = self.total_generations - algorithm.n_gen
            est_remaining = avg_time_per_gen * remaining_gens

            print(f"| {algorithm.n_gen}/{self.total_generations} | {elapsed:.1f}s | {est_remaining:.1f}s |")


if __name__ == '__main__':
    num_targets = 20
    num_generations = 30
    population_size = 100
    num_offsprings = int(population_size / 2)
    min_joints, max_joints = 4, 7
    callback = TimeTrackingCallback(num_generations)

    # Define target end-effector positions.
    target_positions = np.random.uniform(low=[-2.0, 0, 0], high=[2.0, 2.0, 2.0], size=(num_targets, 3)).tolist()

    problem = KinematicChainProblem(
        target_positions, 
        min_joints=min_joints, 
        max_joints=max_joints,
        alpha=1.0,
        beta=1.0,
        delta=1.0,
        gamma=1.0)

    # Configure the NSGA-II algorithm.
    algorithm = NSGA2(
        pop_size=population_size,  
        sampling=LatinHypercubeSampling(), 
        crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),  
        mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20), 
        n_offsprings=num_offsprings, 
        eliminate_duplicates=True
    )
    
    # Run the optimization.
    res = minimize(
        problem, 
        algorithm, 
        termination=('n_gen', num_generations), 
        seed=1, 
        verbose=False,
        callback=callback
    )

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
    filename = f"{storage_dir}nsga2_results_rrmc_no_obj_scale.pkl"

    # Save results to a pickle file
    with open(filename, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Results saved to {filename}")
