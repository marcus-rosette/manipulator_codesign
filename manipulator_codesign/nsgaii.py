import numpy as np
from scipy.spatial.transform import Rotation
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

from manipulator_codesign.kinematic_chain import KinematicChainRTB, KinematicChainPyBullet


def decode_decision_vector(x, min_joints=2, max_joints=5):
    """
    Decode a decision vector into kinematic chain parameters.
    
    The decision vector is encoded as follows:
      - x[0]: Number of joints (integer in [min_joints, max_joints])
      - For each of max_joints potential joints (i = 0,...,max_joints-1):
          x[3*i + 1]: Joint type (integer 0 or 1)
          x[3*i + 2]: Joint axis (integer 0, 1, or 2 corresponding to 'x','y','z')
          x[3*i + 3]: Link length (continuous value in [0.1, 0.75])
          
    Only the first 'num_joints' (decoded from x[0]) are used.
    
    Returns:
        num_joints (int): Actual number of joints (clipped between min_joints and max_joints).
        joint_types (list of int): List of joint types for the active joints.
        joint_axes (list of str): List of joint axes ('x','y','z') for the active joints.
        link_lengths (list of float): List of link lengths for the active joints.
    """
    # Decode number of joints and clip to [min_joints, max_joints]
    num_joints = int(np.rint(x[0]))
    num_joints = max(min_joints, min(num_joints, max_joints))
    
    joint_types = []
    joint_axes = []
    link_lengths = []
    axis_map = {0: 'x', 1: 'y', 2: 'z'}
    
    # Loop through each active joint and decode its parameters.
    for i in range(num_joints):
        idx = 3 * i + 1  # Starting index for joint i's parameters
        
        # Decode joint type and ensure it's either 0 or 1
        joint_type = int(np.rint(x[idx]))
        joint_type = 0 if joint_type < 0 else (1 if joint_type > 1 else joint_type)
        joint_types.append(joint_type)
        
        # Decode joint axis and map to a character ('x', 'y', 'z')
        joint_axis = int(np.rint(x[idx + 1]))
        joint_axis = 0 if joint_axis < 0 else (2 if joint_axis > 2 else joint_axis)
        joint_axes.append(axis_map[joint_axis])
        
        # Decode link length and clip within allowed bounds [0.1, 0.75]
        length = np.clip(x[idx + 2], 0.1, 0.75)
        link_lengths.append(length)
    
    return num_joints, joint_types, joint_axes, link_lengths


class KinematicChainProblem(ElementwiseProblem):
    def __init__(self, target_positions, backend='pybullet', renders=False,
                 min_joints=2, max_joints=5):
        """
        Define the optimization problem for kinematic chain design.
        
        The decision vector is parameterized by the number of joints and
        joint parameters. Its length is:
            1 + (max_joints * 3)
        where:
          - x[0] is the number of joints (integer in [min_joints, max_joints])
          - For each of max_joints joints:
              x[3*i+1]: Joint type (0 or 1)
              x[3*i+2]: Joint axis (0,1,2 corresponding to 'x','y','z')
              x[3*i+3]: Link length (continuous between 0.1 and 0.75)
        
        Objectives:
          1. Minimize overall tracking error (e.g., maximum error among targets).
          2. Minimize the number of joints (to favor simpler designs).
        """
        self.min_joints = min_joints
        self.max_joints = max_joints

        # Build lower and upper bounds for the decision vector.
        # First element: number of joints.
        lower_bound = [min_joints]
        upper_bound = [max_joints]
        # For each potential joint (max_joints in total)
        for _ in range(max_joints):
            lower_bound.extend([0, 0, 0.1])   # Joint type, joint axis, link length lower bounds.
            upper_bound.extend([1, 2, 0.75])    # Joint type, joint axis, link length upper bounds.
        
        n_var = 1 + (max_joints * 3)
        super().__init__(n_var=n_var, n_obj=2, n_constr=0,
                         xl=np.array(lower_bound), xu=np.array(upper_bound))
        self.target_positions = np.array(target_positions)
        self.backend = backend

        if self.backend == 'pybullet':
            # For PyBullet, initialize connection and related objects.
            from pyb_utils import PybUtils
            from load_objects import LoadObjects
            self.pyb = PybUtils(self, renders=renders)
            self.object_loader = LoadObjects(self.pyb.con)
        
    def _chain_factory(self, num_joints, joint_types, joint_axes, link_lengths):
        """
        Create a kinematic chain instance using the chosen backend.
        """
        if self.backend == 'pybullet':
            return KinematicChainPyBullet(self.pyb.con, num_joints, joint_types, joint_axes, link_lengths)
        else:
            return KinematicChainRTB(num_joints, joint_types, joint_axes, link_lengths)

    def _evaluate(self, x, out, *args, **kwargs):
        # Decode the decision vector (pass along min_joints and max_joints).
        num_joints, joint_types, joint_axes, link_lengths = decode_decision_vector(
            x, self.min_joints, self.max_joints)
        
        # Create the kinematic chain.
        chain = self._chain_factory(num_joints, joint_types, joint_axes, link_lengths)
        
        # Compute the error for each target.
        target_errors = np.array([chain.compute_fitness(target) for target in self.target_positions])
        
        # Option 1: Use the maximum error among targets as the overall error.
        overall_error = target_errors.max()

        # overall_error = chain.compute_motion_plan_fitness(self.target_positions)

        # Option 2: Use the mean error across targets as the overall error.
        # overall_error = target_errors.mean()
        
        # Two objectives: overall tracking error and number of joints.
        out["F"] = [overall_error, num_joints]


if __name__ == '__main__':
    # Define target end-effector positions.
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

    target_orientations = [
        Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [-90, 90, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_quat(),
        Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat()
    ]
    target_poses = list(zip(target_positions, target_orientations))
    
    # You can update these variables to search over a different range of joints.
    min_joints = 2
    max_joints = 7
    
    # Instantiate the problem using the updated joint limits.
    problem = KinematicChainProblem(target_poses, min_joints=min_joints, max_joints=max_joints)
    
    # Configure the NSGA-II algorithm.
    algorithm = NSGA2(pop_size=30, eliminate_duplicates=True)
    
    # Run the optimization for 20 generations.
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 10),
                   seed=1,
                   verbose=True)
    
    # Identify the best solution based on the smallest tracking error.
    best_idx = np.argmin(res.F[:, 0])
    best_decision_vector = res.X[best_idx]
    
    # Decode the best decision vector using the same joint limits.
    num_joints, joint_types, joint_axes, link_lengths = decode_decision_vector(
        best_decision_vector, min_joints, max_joints)
    
    # Create the best kinematic chain.
    best_chain = KinematicChainRTB(num_joints, joint_types, joint_axes, link_lengths,
                                   robot_name="NSGA_robot", save_urdf_dir=None)
    
    # Generate and save the URDF.
    best_chain.create_urdf()
    best_chain.save_urdf('best_chain')
    
    # Print the best chain description and objectives.
    best_chain.describe()
    print("\nBest solution objectives:")
    print(f"  Tracking Error: {res.F[best_idx, 0]}")
    print(f"  Number of Joints: {res.F[best_idx, 1]}")
    
    # Visualize the Pareto front.
    Scatter().add(res.F).show()
