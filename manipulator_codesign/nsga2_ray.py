import numpy as np
import ray
from scipy.spatial.transform import Rotation
import pickle
import time
import os
from datetime import datetime

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.load_objects import LoadObjects

# Remote worker to evaluate one individual using direct PyBullet connection
@ray.remote
def evaluate_individual(x, target_positions, min_joints, max_joints, alpha, beta, delta, gamma, renders=False):
    # Direct import of PyBullet and data path
    import pybullet as p
    import pybullet_data
    # Establish a DIRECT-mode connection for each process
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    # Use the module as the connection interface
    con = p

    # Optionally load environment objects
    _ = LoadObjects(con)

    # Decode decision vector
    num_joints, joint_types, joint_axes, link_lengths = decode_decision_vector(
        x, min_joints, max_joints)

    # Build kinematic chain
    chain = KinematicChainPyBullet(con, num_joints, joint_types, joint_axes, link_lengths)
    if not chain.is_built:
        chain.build_robot()
    chain.load_robot()

    # Compute metrics
    chain.compute_chain_metrics(target_positions)

    # Collect results
    metrics = {
        'pose_error': chain.mean_pose_error,
        'torque': chain.mean_torque,
        'joint_count': chain.num_joints,
        'conditioning_index': chain.global_conditioning_index,
        'manip_score_rrmc': chain.mean_manip_score_rrmc,
        'delta_joint_score_rrmc': chain.mean_delta_joint_score_rrmc,
        'pos_error_rrmc': chain.mean_pos_error_rrmc,
        'ori_error_rrmc': chain.mean_ori_error_rrmc
    }

    # Clean up the simulation
    con.resetSimulation()
    p.disconnect(client)

    return metrics


class KinematicChainProblem(Problem):
    def __init__(self, target_positions, backend='pybullet', renders=False,
                 min_joints=2, max_joints=5,
                 alpha=10.0, beta=0.1, delta=0.1, gamma=3.0):
        self.target_positions = np.array(target_positions)
        self.min_joints = min_joints
        self.max_joints = max_joints
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.n_obj = 6

        # Decision vector bounds: [num_joints] + [type, axis, length] * max_joints
        xl = [min_joints] + [0, 0, 0.1] * max_joints
        xu = [max_joints] + [1, 2, 0.75] * max_joints
        super().__init__(n_var=len(xl), n_obj=self.n_obj, n_constr=0,
                         xl=np.array(xl), xu=np.array(xu))

    def _evaluate(self, X, out, *args, **kwargs):
        n_ind = X.shape[0]
        # Initialize storage
        raw_pose_errors = np.zeros(n_ind)
        raw_torques = np.zeros(n_ind)
        joint_counts = np.zeros(n_ind)
        conditioning_indices = np.zeros(n_ind)
        manip_scores_rrmc = np.zeros(n_ind)
        delta_joint_score_rrmc = np.zeros(n_ind)
        raw_pos_errors_rrmc = np.zeros(n_ind)
        raw_ori_errors_rrmc = np.zeros(n_ind)

        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
            print("Resources:", ray.available_resources())

        # Dispatch parallel tasks
        futures = [
            evaluate_individual.remote(
                X[i, :], self.target_positions,
                self.min_joints, self.max_joints,
                self.alpha, self.beta, self.delta, self.gamma
            ) for i in range(n_ind)
        ]
        results = ray.get(futures)

        # Unpack
        for i, m in enumerate(results):
            raw_pose_errors[i] = m['pose_error']
            raw_torques[i] = m['torque']
            joint_counts[i] = m['joint_count']
            conditioning_indices[i] = m['conditioning_index']
            manip_scores_rrmc[i] = m['manip_score_rrmc']
            delta_joint_score_rrmc[i] = m['delta_joint_score_rrmc']
            raw_pos_errors_rrmc[i] = m['pos_error_rrmc']
            raw_ori_errors_rrmc[i] = m['ori_error_rrmc']

        # Normalize
        pose_error_norm = self.tanh_normalize_to_01(raw_pose_errors)
        torque_norm = self.tanh_normalize_to_01(raw_torques)
        delta_joint_norm = self.tanh_normalize_to_01(delta_joint_score_rrmc)
        pos_error_norm = self.tanh_normalize_to_01(raw_pos_errors_rrmc)

        # Compute objectives
        F = np.zeros((n_ind, self.n_obj))
        for i in range(n_ind):
            f1 = self.alpha * pose_error_norm[i]
            f2 = self.beta * torque_norm[i]
            f3 = self.delta * (joint_counts[i] / self.max_joints)
            f4 = self.gamma * abs(conditioning_indices[i] - 1)
            f5 = delta_joint_norm[i]
            f6 = pos_error_norm[i]
            F[i, :] = [f1, f2, f3, f4, f5, f6]
        out['F'] = F

    @staticmethod
    def tanh_normalize_to_01(data):
        mean = np.mean(data)
        std = np.std(data) if np.std(data) != 0 else 1e-8
        return (1 + np.tanh((data - mean) / std)) / 2


class TimeTrackingCallback:
    def __init__(self, total_generations):
        self.total_generations = total_generations
        self.start_time = None

    def __call__(self, algorithm):
        if algorithm.n_gen == 1:
            self.start_time = time.time()
            print(f"Starting generation 1/{self.total_generations}")
        else:
            elapsed = time.time() - self.start_time
            avg = elapsed / (algorithm.n_gen - 1)
            remain = avg * (self.total_generations - algorithm.n_gen)
            print(f"Gen {algorithm.n_gen}/{self.total_generations} - Elapsed: {elapsed:.1f}s, Remain: {remain:.1f}s")


if __name__ == '__main__':
    # Init Ray
    ray.init(num_cpus=os.cpu_count())

    # Experiment params
    num_targets = 10
    num_generations = 10
    population_size = 100
    num_offsprings = population_size // 2
    min_joints, max_joints = 4, 7

    target_positions = np.random.uniform(
        low=[-2.0, 0, 0], high=[2.0, 2.0, 2.0], size=(num_targets, 3)
    ).tolist()

    problem = KinematicChainProblem(
        target_positions, min_joints=min_joints, max_joints=max_joints,
        alpha=1.0, beta=1.0, delta=1.0, gamma=1.0
    )

    algorithm = NSGA2(
        pop_size=population_size,
        sampling=LatinHypercubeSampling(),
        crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
        mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20),
        n_offsprings=num_offsprings,
        eliminate_duplicates=True
    )

    callback = TimeTrackingCallback(num_generations)
    res = minimize(
        problem, algorithm,
        termination=('n_gen', num_generations),
        seed=1, verbose=False, callback=callback
    )

    results_dict = {
        'decision_vecs': res.X, 'objective_vals': res.F,
        'min_joints': min_joints, 'max_joints': max_joints,
        'weights': (problem.alpha, problem.beta, problem.delta, problem.gamma)
    }
    storage_dir = 'C:/Users/marcu/OneDrive/Documents/GitHub/manipulator_codesign/data/nsga2_results/'
    os.makedirs(storage_dir, exist_ok=True)
    filename = os.path.join(storage_dir, f"nsga2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {filename}")
