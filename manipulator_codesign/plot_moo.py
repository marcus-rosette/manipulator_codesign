import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
import numpy as np
import pickle

from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.pyb_utils import PybUtils
from manipulator_codesign.load_objects import LoadObjects
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet

# PARAMS
plot_pareto = False
renders = False
create_urdf = True
results_file = 'nsga2_results.pkl'
ideal = np.array([0.5, 0.0, 0.0, 3.0]) # Define the ideal objective vector

# LOAD DATA
storage_dir = '/home/marcus/IMML/manipulator_codesign/data/nsga2_results/'
with open(f"{storage_dir}{results_file}", "rb") as f:
    results_dict = pickle.load(f)

# Extract the decision vectors and objective values
res_X = results_dict['decision_vecs']
res_F = results_dict['objective_vals']
min_joints = results_dict['min_joints']
max_joints = results_dict['max_joints']


if create_urdf:
    pyb = PybUtils(renders=False)
    object_loader = LoadObjects(pyb.con)

    # Compute the Euclidean distance from each candidate's objective vector to the ideal.
    distances = np.linalg.norm(res_F - ideal, axis=1)

    # Find the candidate with the smallest distance to the ideal.
    best_idx = np.argmin(distances)
    best_decision_vector = res_X[best_idx]

    num_joints, joint_types, joint_axes, link_lengths = decode_decision_vector(best_decision_vector, min_joints, max_joints)
    best_chain = KinematicChainPyBullet(pyb.con, num_joints, joint_types, joint_axes, link_lengths, robot_name="NSGA_robot_balanced")

    best_chain.create_urdf()
    best_chain.save_urdf('best_chain_balanced')

if plot_pareto:
    plot = Scatter()
    plot.add(res_F)
    plot.axis_labels = ["Pose Error", "Torque Penalty", "Joint Penalty", "Conditioning Index"]
    plot.show()

     # Explicitly close all matplotlib figures to prevent pymoo plt.close() error
    del plot
    plt.close('all')