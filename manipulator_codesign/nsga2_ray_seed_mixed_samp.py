import numpy as np
import ray
import pickle
import os
from datetime import datetime

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.urdf_to_decision_vector import encode_seed, urdf_to_decision_vector
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.load_objects import LoadObjects

# -------- Seeded Sampling --------
class SeededSampling(Sampling):
    def __init__(self, var_types, seeds, fallback_sampler):
        super().__init__()
        self.var_types = var_types
        self.seeds = [np.asarray(x, dtype=float) for x in seeds]
        self.fallback = fallback_sampler
        # Verify seed length
        n_var = len(var_types)
        for x in self.seeds:
            assert x.shape == (n_var,), f"Seed must be length {n_var}"

    def _do(self, problem, n_samples, **kwargs):
        print("[SeededSampling] Generating initial population with seeds...")
        n_seeds = min(len(self.seeds), n_samples)
        X_seeded = np.stack(self.seeds[:n_seeds], axis=0)
        n_rem = n_samples - n_seeds
        if n_rem > 0:
            X_rest = self.fallback._do(problem, n_rem, **kwargs)
            return np.vstack([X_seeded, X_rest])
        return X_seeded
    

class MixedSampling(Sampling):
    def __init__(self, var_types):
        super().__init__()
        self.var_types = var_types
        self.lhs = LatinHypercubeSampling()
        self.int_sampler = IntegerRandomSampling()

    def _do(self, problem, n_samples, **kwargs):
        X = self.lhs._do(problem, n_samples, **kwargs)
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
        real_idx = [i for i, t in enumerate(self.var_types) if t == 'real']
        int_idx = [i for i, t in enumerate(self.var_types) if t != 'real']
        Y = np.empty_like(X)
        # SBX on reals
        if real_idx:
            sub_xl = problem.xl[real_idx]
            sub_xu = problem.xu[real_idx]
            sub_prob = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                               xl=sub_xl, xu=sub_xu)
            Yr = self.sbx._do(sub_prob, X[:, :, real_idx], **kwargs)
            for idx, col in enumerate(real_idx):
                Y[:, :, col] = Yr[:, :, idx]
        # Uniform on ints
        if int_idx:
            Yi = self.uni._do(problem, X[:, :, int_idx], **kwargs)
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
        # 2D
        if X.ndim == 2:
            n, _ = X.shape
            Y = X.copy()
            if real_idx:
                sub_xl = problem.xl[real_idx]
                sub_xu = problem.xu[real_idx]
                sub_prob = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                                   xl=sub_xl, xu=sub_xu)
                Yr = PolynomialMutation(eta=self.eta_pm, prob=self.prob_real)._do(
                    sub_prob, X[:, real_idx], **kwargs)
                Y[:, real_idx] = Yr
            for col in int_idx:
                mask = np.random.rand(n) < self.prob_int
                if mask.any():
                    lo, hi = int(problem.xl[col]), int(problem.xu[col])
                    Y[mask, col] = np.random.randint(lo, hi + 1, mask.sum())
            return Y
        # 3D fallback
        n = X.shape[0]
        Y = X.copy()
        if real_idx:
            sub_xl = problem.xl[real_idx]
            sub_xu = problem.xu[real_idx]
            sub_prob = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                               xl=sub_xl, xu=sub_xu)
            Yr = PolynomialMutation(eta=self.eta_pm, prob=self.prob_real)._do(
                sub_prob, X[:, 0, real_idx], **kwargs)
            Y[:, 0, real_idx] = Yr
        for col in int_idx:
            mask = np.random.rand(n) < self.prob_int
            if mask.any():
                lo, hi = int(problem.xl[col]), int(problem.xu[col])
                Y[mask, 0, col] = np.random.randint(lo, hi + 1, mask.sum())
        return Y


# -------- Remote Eval --------
@ray.remote
def evaluate_individual(x, targets, min_j, max_j, alpha, beta, delta, gamma):
    import pybullet as p, pybullet_data
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    _ = LoadObjects(p)
    n, types, axes, lengths = decode_decision_vector(x, min_j, max_j)
    ch = KinematicChainPyBullet(p, n, types, axes, lengths)
    if not ch.is_built:
        ch.build_robot()
    ch.load_robot()
    ch.compute_chain_metrics(targets)
    m = {
        'pose_error': ch.mean_pose_error,
        'torque': ch.mean_torque,
        'joint_count': ch.num_joints,
        'conditioning_index': ch.global_conditioning_index,
        'delta_joint_score_rrmc': ch.mean_delta_joint_score_rrmc,
        'pos_error_rrmc': ch.mean_pos_error_rrmc
    }
    p.resetSimulation()
    p.disconnect(c)
    return m


# -------- Problem --------
class KinematicChainProblem(Problem):
    def __init__(self, targets, min_joints=2, max_joints=7, alpha=1, beta=1, delta=1, gamma=1, cal_samples=15):
        print("[KinematicChainProblem] Initializing problem and calibrating objective bounds...")
        self.targets = np.array(targets)
        self.min_joints, self.max_joints = min_joints, max_joints
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma
        n_obj = 6

        # Decision vector bounds: [num_joints] + [type, axis, length] * max_joints
        xl = [min_joints] + [0, 0, 0.1] * max_joints
        xu = [max_joints] + [2, 2, 0.75] * max_joints

        super().__init__(n_var=len(xl), n_obj=n_obj, xl=np.array(xl), xu=np.array(xu))

        # Calibrate
        self._x_cal = np.random.uniform(self.xl, self.xu, (cal_samples, len(xl)))
        self._parallel_calibration()

    def _parallel_calibration(self):
        print("[Calibration] Running parallel calibration samples...")
        res = ray.get([evaluate_individual.remote(x, self.targets, self.min_joints, self.max_joints,
                                                  self.alpha, self.beta, self.delta, self.gamma)
                       for x in self._x_cal])

        def b(a):
            lo, hi = min(a), max(a)
            return (lo, hi if hi > lo else lo + 1e-6)

        self.pose_bounds = b([r['pose_error'] for r in res])
        self.torque_bounds = b([r['torque'] for r in res])
        self.delta_bounds = b([r['delta_joint_score_rrmc'] for r in res])
        self.pos_bounds = b([r['pos_error_rrmc'] for r in res])
        self.jcount_bounds = b([r['joint_count'] for r in res])

    def _evaluate(self, X, out, *a, **k):
        n = X.shape[0]
        res = ray.get([evaluate_individual.remote(X[i], self.targets, self.min_joints, self.max_joints,
                                                  self.alpha, self.beta, self.delta, self.gamma)
                       for i in range(n)])
        F = np.zeros((n, self.n_obj))

        def lin(v, lo, hi):
            return np.clip((v - lo) / max(1e-8, hi - lo), 0, 1)

        for i, r in enumerate(res):
            p_lo, p_hi = self.pose_bounds
            t_lo, t_hi = self.torque_bounds
            d_lo, d_hi = self.delta_bounds
            pr_lo, pr_hi = self.pos_bounds
            jc_lo, jc_hi = self.jcount_bounds
            F[i, 0] = self.alpha * lin(r['pose_error'], p_lo, p_hi)
            F[i, 1] = self.beta * lin(r['torque'], t_lo, t_hi)
            F[i, 2] = self.delta * lin(r['joint_count'], jc_lo, jc_hi)
            F[i, 3] = self.gamma * abs(r['conditioning_index'] - 1)
            F[i, 4] = lin(r['delta_joint_score_rrmc'], d_lo, d_hi)
            F[i, 5] = lin(r['pos_error_rrmc'], pr_lo, pr_hi)
        out['F'] = F


# -------- Main --------
if __name__ == '__main__':
    ray.init(num_cpus=os.cpu_count())

    # -------- Mixed Operators --------
    max_joints = 7
    var_types = ['int'] + ['int', 'int', 'real'] * max_joints

    # -------- Load URDF Seeds --------
    urdf_dir = 'manipulator_codesign/urdf/robots/nsga2_seeds'
    urdfs = [os.path.join(urdf_dir, f) for f in os.listdir(urdf_dir) if f.endswith('.urdf')]

    # load raw structural seeds...
    raw_seeds = [urdf_to_decision_vector(u) for u in urdfs]

    # encode into numeric decision vectors of length 1+3*max_joints
    seeds = [encode_seed(s, max_joints=max_joints)
            for s in raw_seeds]

    # Set up sampling with seeds
    fallback = MixedSampling(var_types)
    sampling = SeededSampling(var_types, seeds, fallback)
    crossover = MixedCrossover(var_types)
    mutation = MixedMutation(var_types, prob_real=1.0 / (1 + 3 * max_joints), prob_int=0.1)

    targets = np.random.uniform([-2, 0, 0], [2, 2, 2], (5, 3)).tolist()
    problem = KinematicChainProblem(targets)

    algo = NSGA2(pop_size=10, sampling=sampling, crossover=crossover, mutation=mutation, eliminate_duplicates=True) # n_offsprings=16,
    res = minimize(problem, algo, termination=('n_gen', 20), seed=1, verbose=True)

    os.makedirs('data', exist_ok=True)
    fn = os.path.join('data', f'results_{datetime.now():%Y%m%d_%H%M%S}.pkl')
    with open(fn, 'wb') as f:
        pickle.dump({'X': res.X, 'F': res.F}, f)
    print(f"Saved results to {fn}")
