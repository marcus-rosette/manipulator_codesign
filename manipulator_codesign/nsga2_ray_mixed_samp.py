import numpy as np
import ray
import pickle
import os
from datetime import datetime

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
# Core problem
from pymoo.core.problem import Problem
# Operators base
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
# Specific samplers
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling, IntegerRandomSampling
# Crossovers
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ux import UniformCrossover
# Mutation
from pymoo.operators.mutation.pm import PolynomialMutation

from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.load_objects import LoadObjects

# -------- Custom Mixed Operators --------
max_joints = 7
var_types = ['int'] + ['int','int','real'] * max_joints

class MixedSampling(Sampling):
    def __init__(self, var_types):
        super().__init__()
        self.var_types = var_types
        self.lhs = LatinHypercubeSampling()
        self.int_sampler = IntegerRandomSampling()

    def _do(self, problem, n_samples, **kwargs):
        # Continuous via LHS
        X = self.lhs._do(problem, n_samples, **kwargs)
        # Integer genes via uniform integer sampling
        Xi = self.int_sampler._do(problem, n_samples, **kwargs)
        for i, t in enumerate(self.var_types):
            if t != 'real':
                X[:, i] = Xi[:, i]
        return X

class MixedCrossover(Crossover):
    def __init__(self, var_types, eta_sbx=10, prob_real=0.9, prob_int=0.5):
        super().__init__(2, 2)
        self.var_types = var_types
        self.sbx = SimulatedBinaryCrossover(prob=prob_real, eta=eta_sbx)
        self.uni = UniformCrossover(prob=prob_int)

    def _do(self, problem, X, **kwargs):
        real_idx = [i for i,t in enumerate(self.var_types) if t == 'real']
        int_idx  = [i for i,t in enumerate(self.var_types) if t != 'real']
        Y = np.empty_like(X)
        # Crossover real variables via SBX using a sub-problem
        if real_idx:
            # Create a sub-problem for real dims to supply correct bounds
            sub_xl = problem.xl[real_idx]
            sub_xu = problem.xu[real_idx]
            sub_problem = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                                   xl=sub_xl, xu=sub_xu)
            Xr = X[:, :, real_idx]
            Yr = self.sbx._do(sub_problem, Xr, **kwargs)
            for idx, col in enumerate(real_idx):
                Y[:, :, col] = Yr[:, :, idx]
        # Crossover integer variables via Uniform
        if int_idx:
            Xi = X[:, :, int_idx]
            Yi = self.uni._do(problem, Xi, **kwargs)
            for idx, col in enumerate(int_idx):
                Y[:, :, col] = Yi[:, :, idx].round().astype(int)
        return Y

class MixedMutation(Mutation):
    def __init__(self, var_types, eta_pm=20, prob_real=None, prob_int=0.1):
        super().__init__(1, 1)
        self.var_types = var_types
        self.eta_pm = eta_pm
        self.prob_real = prob_real
        self.prob_int = prob_int

    def _do(self, problem, X, **kwargs):
        real_idx = [i for i, t in enumerate(self.var_types) if t == 'real']
        int_idx = [i for i, t in enumerate(self.var_types) if t != 'real']

        def mutate_reals(Xr):
            sub_xl = problem.xl[real_idx]
            sub_xu = problem.xu[real_idx]
            sub_problem = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                                   xl=sub_xl, xu=sub_xu)
            pm = PolynomialMutation(eta=self.eta_pm, prob=self.prob_real)
            return pm._do(sub_problem, Xr, **kwargs)

        # Handle 2D input: shape (n_ind, n_var)
        if X.ndim == 2:
            n_ind, _ = X.shape
            Y = X.copy()
            if real_idx:
                Xr = Y[:, real_idx]
                Yr = mutate_reals(Xr)
                Y[:, real_idx] = Yr
            for col in int_idx:
                mask = np.random.rand(n_ind) < self.prob_int
                if mask.any():
                    lo, hi = int(problem.xl[col]), int(problem.xu[col])
                    Y[mask, col] = np.random.randint(lo, hi + 1, size=mask.sum())
            return Y

        # Handle 3D input: shape (n_ind, 1, n_var)
        n_ind = X.shape[0]
        Y = X.copy()
        if real_idx:
            Xr = Y[:, 0, real_idx]
            Yr = mutate_reals(Xr)
            Y[:, 0, real_idx] = Yr
        for col in int_idx:
            mask = np.random.rand(n_ind) < self.prob_int
            if mask.any():
                lo, hi = int(problem.xl[col]), int(problem.xu[col])
                Y[mask, 0, col] = np.random.randint(lo, hi + 1, size=mask.sum())
        return Y

# Instantiate operators
sampling = MixedSampling(var_types)
crossover = MixedCrossover(var_types)
mutation  = MixedMutation(var_types, prob_real=1.0/(1 + 3*max_joints), prob_int=0.1)

# -------- Remote Evaluation --------
@ray.remote
def evaluate_individual(x, target_positions,
                             min_joints, max_joints,
                             alpha, beta, delta, gamma):
    import pybullet as p, pybullet_data
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    _ = LoadObjects(p)
    n, types, axes, lengths = decode_decision_vector(x, min_joints, max_joints)
    chain = KinematicChainPyBullet(p, n, types, axes, lengths)
    if not chain.is_built:
        chain.build_robot()
    chain.load_robot()
    chain.compute_chain_metrics(target_positions)
    metrics = {
        'pose_error': chain.mean_pose_error,
        'torque': chain.mean_torque,
        'joint_count': chain.num_joints,
        'conditioning_index': chain.global_conditioning_index,
        'delta_joint_score_rrmc': chain.mean_delta_joint_score_rrmc,
        'pos_error_rrmc': chain.mean_pos_error_rrmc
    }
    p.resetSimulation()
    p.disconnect(client)
    return metrics

# -------- Problem Definition --------
class KinematicChainProblem(Problem):
    def __init__(self, target_positions,
                 min_joints=2, max_joints=7,
                 alpha=1.0, beta=1.0, delta=1.0, gamma=1.0,
                 calibrate_samples=15):
        self.targets = np.array(target_positions)
        self.min_joints, self.max_joints = min_joints, max_joints
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma
        n_obj = 6
        xl = [min_joints] + [0, 0, 0.1] * max_joints
        xu = [max_joints] + [1, 2, 0.75] * max_joints
        super().__init__(n_var=len(xl), n_obj=n_obj, xl=np.array(xl), xu=np.array(xu))

        # normalization bounds
        self.pose_bounds = (np.inf, -np.inf)
        self.torque_bounds = (np.inf, -np.inf)
        self.jcount_bounds = (min_joints, max_joints)
        self.delta_bounds = (np.inf, -np.inf)
        self.pos_bounds = (np.inf, -np.inf)

        # calibrate in parallel
        # prepare random samples
        self._cal_samples = calibrate_samples
        self._x_cal = np.random.uniform(self.xl, self.xu,
                                        (calibrate_samples, len(xl)))
        self._parallel_calibration()

    def _parallel_calibration(self):
        # launch all remote calls
        futures = [evaluate_individual.remote(
            x, self.targets,
            self.min_joints, self.max_joints,
            self.alpha, self.beta, self.delta, self.gamma
        ) for x in self._x_cal]
        results = ray.get(futures)
        # collect metrics
        pose = [r['pose_error'] for r in results]
        torque = [r['torque'] for r in results]
        jcount = [r['joint_count'] for r in results]
        delta = [r['delta_joint_score_rrmc'] for r in results]
        pos = [r['pos_error_rrmc'] for r in results]
        # compute bounds
        def bounds(arr): lo, hi = min(arr), max(arr); return (lo, hi if hi>lo else lo+1e-6)
        self.pose_bounds = bounds(pose)
        self.torque_bounds = bounds(torque)
        self.delta_bounds = bounds(delta)
        self.pos_bounds = bounds(pos)

    def _evaluate(self, X, out, *args, **kwargs):
        n_ind = X.shape[0]
        if not ray.is_initialized():
            ray.init()
        futures = [evaluate_individual.remote(
            X[i], self.targets,
            self.min_joints, self.max_joints,
            self.alpha, self.beta, self.delta, self.gamma
        ) for i in range(n_ind)]
        results = ray.get(futures)
        # build objective matrix
        F = np.zeros((n_ind, self.n_obj))
        def lin(a, lo, hi):
            return np.clip((a - lo) / max(1e-8, hi - lo), 0.0, 1.0)
        for i, m in enumerate(results):
            p_lo, p_hi = self.pose_bounds
            t_lo, t_hi = self.torque_bounds
            d_lo, d_hi = self.delta_bounds
            pr_lo, pr_hi = self.pos_bounds
            F[i, 0] = self.alpha * lin(m['pose_error'], p_lo, p_hi)
            F[i, 1] = self.beta  * lin(m['torque'], t_lo, t_hi)
            F[i, 2] = self.delta * lin(m['joint_count'], *self.jcount_bounds)
            F[i, 3] = self.gamma * abs(m['conditioning_index'] - 1)
            F[i, 4] = lin(m['delta_joint_score_rrmc'], d_lo, d_hi)
            F[i, 5] = lin(m['pos_error_rrmc'], pr_lo, pr_hi)
        out['F'] = F

# -------- Run NSGA-II --------
if __name__ == '__main__':
    ray.init(num_cpus=os.cpu_count())
    targets = np.random.uniform([-2,0,0], [2,2,2], (10,3)).tolist()
    problem = KinematicChainProblem(targets)

    algorithm = NSGA2(
        pop_size=15,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )

    res = minimize(
        problem, algorithm,
        termination=('n_gen', 10),
        seed=1, verbose=True
    )

    os.makedirs('data', exist_ok=True)
    fname = os.path.join('data', f'nsga2_mixed_{datetime.now():%Y%m%d_%H%M%S}.pkl')
    with open(fname, 'wb') as f:
        pickle.dump({'X': res.X, 'F': res.F}, f)
    print(f"Results saved to {fname}")