import os
import numpy as np
import pickle
import ray
import pybullet as p
import pybullet_data
import time

from datetime import datetime
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.load_objects import LoadObjects

# Initialize Ray
# ray.init(ignore_reinit_error=True)
@ray.remote
def evaluate_individual(x, target_positions, min_j, max_j):
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    _ = LoadObjects(p)

    nj, types, axes, lengths = decode_decision_vector(x, min_j, max_j)
    chain = KinematicChainPyBullet(p, nj, types, axes, lengths)
    if not chain.is_built:
        chain.build_robot()
    chain.load_robot()
    chain.compute_chain_metrics(target_positions)

    metrics = {
        'pose':       chain.mean_pose_error,
        'torque':     chain.mean_torque,
        'jcount':     chain.num_joints,
        'cond':       abs(chain.global_conditioning_index - 1.0),
        'delta_rrmc': chain.mean_delta_joint_score_rrmc,
        'pos_rrmc':   chain.mean_pos_error_rrmc
    }
    p.resetSimulation()
    p.disconnect(client)
    return metrics

# Mask generator for mixed variables
def make_mask(min_j, max_j):
    mask = ['int']
    for _ in range(max_j):
        mask += ['int', 'int', 'real']
    return mask

class MixedSampling(Sampling):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self.real_sampler = LatinHypercubeSampling()

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))
        X_real = self.real_sampler._do(problem, n_samples)
        for j, m in enumerate(self.mask):
            if m == 'real':
                X[:, j] = X_real[:, j]
            else:
                lo, hi = int(problem.xl[j]), int(problem.xu[j])
                X[:, j] = np.random.randint(lo, hi + 1, size=n_samples)
        return X

class MixedSBX(Crossover):
    def __init__(self, mask, prob=0.9, eta=15):
        super().__init__(2, 2)
        self.mask = mask
        self.sbx  = SimulatedBinaryCrossover(prob=prob, eta=eta)

    def _do(self, problem, X, **kwargs):
        _, n_off, _ = X.shape
        Y = np.zeros_like(X)
        real_idx = [i for i, m in enumerate(self.mask) if m == 'real']

        # Auxiliary problem for continuous part
        class AuxP: pass
        aux = AuxP()
        aux.xl = problem.xl[real_idx]
        aux.xu = problem.xu[real_idx]
        aux.n_var = len(real_idx)

        # SBX on continuous variables
        X_real = X[:, :, real_idx]
        Y_real = self.sbx._do(aux, X_real)
        Y[:, :, real_idx] = Y_real

        # Uniform crossover for discrete variables
        for k in range(n_off):
            for j, m in enumerate(self.mask):
                if m != 'real':
                    if np.random.rand() < 0.5:
                        Y[0, k, j], Y[1, k, j] = X[0, k, j], X[1, k, j]
                    else:
                        Y[0, k, j], Y[1, k, j] = X[1, k, j], X[0, k, j]
        return Y

class MixedMutation(Mutation):
    def __init__(self, mask, eta=20):
        super().__init__()
        self.mask = mask
        self.pm   = PolynomialMutation(prob=None, eta=eta)

    def _do(self, problem, X, **kwargs):
        Xm = X.copy()
        real_idx = [i for i, m in enumerate(self.mask) if m == 'real']
        if real_idx:
            # Auxiliary problem for continuous mutation
            class AuxP: pass
            aux = AuxP()
            aux.xl = problem.xl[real_idx]
            aux.xu = problem.xu[real_idx]
            aux.n_var = len(real_idx)
            Xm[:, real_idx] = self.pm._do(aux, X[:, real_idx])
        # Uniform mutation for discrete variables
        for i in range(X.shape[0]):
            for j, m in enumerate(self.mask):
                if m != 'real' and np.random.rand() < 1.0/problem.n_var:
                    lo, hi = int(problem.xl[j]), int(problem.xu[j])
                    Xm[i, j] = np.random.randint(lo, hi + 1)
        return Xm

class KinematicChainProblem(Problem):
    def __init__(self, target_positions, min_j, max_j,
                 alpha=1.0, beta=1.0, delta=1.0, gamma=1.0,
                 norm_bounds=None):
        self.targets = np.array(target_positions)
        self.min_j, self.max_j = min_j, max_j
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma
        self.n_obj = 6
        self.mask  = make_mask(min_j, max_j)

        xl = [min_j] + [0, 0, 0.1] * max_j
        xu = [max_j] + [1, 2, 0.75] * max_j
        super().__init__(n_var=len(xl), n_obj=self.n_obj, n_constr=0,
                         xl=np.array(xl), xu=np.array(xu))

        self.norm_bounds = norm_bounds or {
            'pose':       (1e-6, 10.0),
            'torque':     (1e-6, 5.0),
            'jcount':     (min_j, max_j),
            'delta_rrmc': (1e-6, 1.0),
            'pos_rrmc':   (1e-6, 1.0)
        }

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        pe = np.zeros(n); tq = np.zeros(n); jc = np.zeros(n)
        ci = np.zeros(n); dr = np.zeros(n); pr = np.zeros(n)
        futures = [evaluate_individual.remote(
            X[i], self.targets, self.min_j, self.max_j) for i in range(n)]
        results = ray.get(futures)
        for i, m in enumerate(results):
            pe[i] = m['pose'];    tq[i] = m['torque']
            jc[i] = m['jcount'];  ci[i] = m['cond']
            dr[i] = m['delta_rrmc']; pr[i] = m['pos_rrmc']

        def norm(a, lo, hi): return np.clip((a - lo)/max(1e-8, hi - lo), 0, 1)
        b = self.norm_bounds
        Np = norm(pe, *b['pose']);   Nt = norm(tq, *b['torque'])
        Nj = norm(jc, *b['jcount']); Nd = norm(dr, *b['delta_rrmc'])
        Nr = norm(pr, *b['pos_rrmc'])

        F = np.zeros((n, self.n_obj))
        for i in range(n):
            F[i,0] = self.alpha * Np[i]
            F[i,1] = self.beta  * Nt[i]
            F[i,2] = self.delta * Nj[i]
            F[i,3] = self.gamma * ci[i]
            F[i,4] = Nd[i]
            F[i,5] = Nr[i]
        out['F'] = F

if __name__ == '__main__':
    targets = np.random.uniform([-2,0,0],[2,2,2],(10,3)).tolist()
    min_j, max_j = 4, 7
    pop, gens = 5, 2
    problem   = KinematicChainProblem(targets, min_j, max_j)
    sampling  = MixedSampling(problem.mask)
    crossover = MixedSBX(problem.mask)
    mutation  = MixedMutation(problem.mask)
    algo = NSGA2(
        pop_size=pop, sampling=sampling,
        crossover=crossover, mutation=mutation,
        eliminate_duplicates=True
    )
    res = minimize(problem, algo, ('n_gen', gens), seed=42, verbose=False)
    out = {'X': res.X, 'F': res.F}
    path = 'data/pareto_mixed_py6113.pkl'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    print(f"Results saved to {path}")