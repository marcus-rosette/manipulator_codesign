import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import ray
import wandb
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.core.callback import Callback 

from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.urdf_to_decision_vector import encode_seed, urdf_to_decision_vector
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.load_objects import LoadObjects


# -------- Seeded & Mixed Operators --------
class SeededSampling(Sampling):
    def __init__(self, var_types, seeds, fallback_sampler):
        super().__init__()
        self.var_types = var_types
        self.seeds = [np.asarray(x, dtype=float) for x in seeds]
        self.fallback = fallback_sampler
        n_var = len(var_types)
        for x in self.seeds:
            assert x.shape == (n_var,)
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

    def _do(self, problem, n_samples, **kwargs):
        # 1) first generate the “real” vars
        X = self.lhs._do(problem, n_samples, **kwargs)

        # 2) now overwrite all integer slots with true randint [xl, xu] inclusive
        for i, t in enumerate(self.var_types):
            if t != 'real':
                lo, hi = int(problem.xl[i]), int(problem.xu[i])
                # +1 on hi to make it inclusive
                X[:, i] = np.random.randint(lo, hi + 1, size=n_samples)

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
        if real_idx:
            sub_prob = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                               xl=problem.xl[real_idx], xu=problem.xu[real_idx])
            Yr = self.sbx._do(sub_prob, X[:, :, real_idx], **kwargs)
            for idx, col in enumerate(real_idx):
                Y[:, :, col] = Yr[:, :, idx]
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
        Y = X.copy()
        if X.ndim == 2:
            if real_idx:
                sub_prob = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                                   xl=problem.xl[real_idx], xu=problem.xu[real_idx])
                Yr = PolynomialMutation(eta=self.eta_pm, prob=self.prob_real)._do(
                    sub_prob, X[:, real_idx], **kwargs)
                Y[:, real_idx] = Yr
            for col in int_idx:
                mask = np.random.rand(Y.shape[0]) < self.prob_int
                if mask.any():
                    lo, hi = int(problem.xl[col]), int(problem.xu[col])
                    Y[mask, col] = np.random.randint(lo, hi + 1, mask.sum())
        else:
            # 3D fallback
            n = X.shape[0]
            if real_idx:
                sub_prob = Problem(n_var=len(real_idx), n_obj=problem.n_obj,
                                   xl=problem.xl[real_idx], xu=problem.xu[real_idx])
                Yr = PolynomialMutation(eta=self.eta_pm, prob=self.prob_real)._do(
                    sub_prob, X[:, 0, real_idx], **kwargs)
                Y[:, 0, real_idx] = Yr
            for col in int_idx:
                mask = np.random.rand(n) < self.prob_int
                if mask.any():
                    lo, hi = int(problem.xl[col]), int(problem.xu[col])
                    Y[mask, 0, col] = np.random.randint(lo, hi + 1, mask.sum())
        return Y


# -------- Remote Evaluation --------
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
    metrics = {
        'pose_error': ch.mean_pose_error,
        'torque': ch.mean_torque,
        'joint_count': ch.num_joints,
        'conditioning_index': ch.global_conditioning_index,
        'delta_joint_score_rrmc': ch.mean_delta_joint_score_rrmc,
        'pos_error_rrmc': ch.mean_pos_error_rrmc
    }
    p.resetSimulation()
    p.disconnect(c)
    return metrics


# -------- Problem Definition --------
class KinematicChainProblem(Problem):
    def __init__(self, targets, min_joints=2, max_joints=7,
                 alpha=1, beta=1, delta=1, gamma=1, cal_samples=15):
        print("[KinematicChainProblem] Initializing and calibrating...")
        self.targets = np.array(targets)
        self.min_joints, self.max_joints = min_joints, max_joints
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma
        n_obj = 6

        xl = [min_joints] + [0, 0, 0.1] * max_joints
        xu = [max_joints] + [2, 2, 0.75] * max_joints
        super().__init__(n_var=len(xl), n_obj=n_obj,
                         xl=np.array(xl), xu=np.array(xu))

        # calibration
        self._x_cal = np.random.uniform(self.xl, self.xu,
                                        (cal_samples, len(xl)))
        self._parallel_calibration()

    def _parallel_calibration(self):
        print("[Calibration] Running parallel calibration samples...")
        res = ray.get([
            evaluate_individual.remote(x, self.targets,
                                       self.min_joints, self.max_joints,
                                       self.alpha, self.beta,
                                       self.delta, self.gamma)
            for x in self._x_cal
        ])
        def bounds(arr):
            lo, hi = min(arr), max(arr)
            return (lo, hi if hi > lo else lo + 1e-6)
        self.pose_bounds    = bounds([r['pose_error'] for r in res])
        self.torque_bounds  = bounds([r['torque'] for r in res])
        self.delta_bounds   = bounds([r['delta_joint_score_rrmc'] for r in res])
        self.pos_bounds     = bounds([r['pos_error_rrmc'] for r in res])
        self.jcount_bounds  = bounds([r['joint_count'] for r in res])

    def _evaluate(self, X, out, *args, **kwargs):
        res = ray.get([
            evaluate_individual.remote(X[i], self.targets,
                                       self.min_joints, self.max_joints,
                                       self.alpha, self.beta,
                                       self.delta, self.gamma)
            for i in range(X.shape[0])
        ])
        F = np.zeros((X.shape[0], self.n_obj))
        def lin(v, lo, hi):
            return np.clip((v - lo) / max(1e-8, hi - lo), 0, 1)
        for i, r in enumerate(res):
            p_lo, p_hi = self.pose_bounds
            t_lo, t_hi = self.torque_bounds
            d_lo, d_hi = self.delta_bounds
            pr_lo, pr_hi = self.pos_bounds
            jc_lo, jc_hi = self.jcount_bounds
            F[i, 0] = self.alpha * lin(r['pose_error'], p_lo, p_hi)
            F[i, 1] = self.beta  * lin(r['torque'], t_lo, t_hi)
            F[i, 2] = self.delta * lin(r['joint_count'], jc_lo, jc_hi)
            F[i, 3] = self.gamma * abs(r['conditioning_index'] - 1)
            F[i, 4] = lin(r['delta_joint_score_rrmc'], d_lo, d_hi)
            F[i, 5] = lin(r['pos_error_rrmc'], pr_lo, pr_hi)
        out["F"] = F


# -------- W&B Callback --------
class WandbLogger(Callback):
    def __init__(self):
        super().__init__()
        self.gen = 0
    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        mean_obj = F.mean(axis=0)
        best_obj = F.min(axis=0)
        
        # objective names
        obj_names = ['pose_error', 'torque', 'joint_count',
                     'conditioning_index', 'delta_joint_score_rrmc',
                     'pos_error_rrmc']

        # log per-generation aggregates
        log_dict = {"generation": self.gen}
        log_dict.update({obj_names[i]: mean_obj[i] for i in range(F.shape[1])})
        log_dict.update({obj_names[i]: best_obj[i] for i in range(F.shape[1])})
        wandb.log(log_dict, step=self.gen)

        self.gen += 1


# -------- Main --------

if __name__ == "__main__":
    ray.init(num_cpus=os.cpu_count())

    # set up operators
    max_joints = 7
    num_generations = 1
    num_population = 5
    num_target_pts = 5
    var_types = ['int'] + ['int','int','real'] * max_joints

    # Find the nsga2_seeds directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_dir = os.path.join(script_dir, 'urdf', 'robots', 'nsga2_seeds')
    urdfs = [os.path.join(urdf_dir, f) for f in os.listdir(urdf_dir) if f.endswith('.urdf')]
    raw_seeds = [urdf_to_decision_vector(u) for u in urdfs]
    seeds     = [encode_seed(s, max_joints=max_joints) for s in raw_seeds]

    fallback  = MixedSampling(var_types)
    sampling  = SeededSampling(var_types, seeds, fallback)
    crossover = MixedCrossover(var_types)
    mutation  = MixedMutation(var_types,
                              prob_real=1.0/(1+3*max_joints),
                              prob_int=0.1)

    targets = np.random.uniform([-2,0,0],[2,2,2],(num_target_pts,3)).tolist()
    problem = KinematicChainProblem(targets)

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None:
        raise RuntimeError("Please set WANDB_API_KEY in your environment")
    wandb.login(key=api_key)

    # Initialize W&B
    wandb.init(
        project="manipulator_codesign",
        entity="rosettem-oregon-state-university",
        name=f"nsga2_run_{datetime.now():%Y%m%d_%H%M%S}",
        config={
            "pop_size": num_population,
            "n_gen": num_generations,
            "min_joints": problem.min_joints,
            "max_joints": problem.max_joints,
            "alpha": problem.alpha,
            "beta": problem.beta,
            "delta": problem.delta,
            "gamma": problem.gamma,
            "sampling": "Seeded+Mixed",
            "crossover": "MixedCrossover",
            "mutation": "MixedMutation"
        }
    )

    callback = WandbLogger()
    algo = NSGA2(pop_size=num_population,
                 sampling=sampling,
                 crossover=crossover,
                 mutation=mutation,
                 eliminate_duplicates=True)

    res = minimize(problem,
                   algo,
                   termination=('n_gen', num_generations),
                   seed=1,
                   verbose=True,
                   callback=callback)

    # save results locally
    os.makedirs('data', exist_ok=True)
    fn = os.path.join('data', f"results_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    with open(fn, 'wb') as f:
        pickle.dump({'X': res.X, 'F': res.F}, f)
    print(f"Saved results to {fn}")

    # log as W&B artifact
    artifact = wandb.Artifact('nsga2-results', type='dataset')
    artifact.add_file(fn)
    wandb.log_artifact(artifact)
    wandb.finish()
