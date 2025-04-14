import argparse
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
import numpy as np
import pickle
import os

from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.pyb_utils import PybUtils
from manipulator_codesign.load_objects import LoadObjects
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet


def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize and analyze NSGA-II optimization results")

    parser.add_argument("-p", "--plot_pareto", action="store_true", help="Plot the Pareto front. Defaults to False")
    parser.add_argument("-r", "--renders", action="store_true", help="Enable rendering in PyBullet simulation. Defaults to False")
    parser.add_argument("-u", "--create_urdf", action="store_true", help="Create and save URDF for best chain. Defaults to False")
    parser.add_argument("-f", "--results_file", type=str, default='nsga2_results.pkl', help="Pickle file with MOO results")
    parser.add_argument("-s", "--storage_dir", type=str, default='/home/marcus/IMML/manipulator_codesign/data/nsga2_results/', help="Directory containing results file")
    parser.add_argument("-i", "--ideal_vector", type=str, default="0.0,0.1,0.0,3.0", help="Comma-separated ideal objective vector (e.g., '0.0,0.1,0.0,3.0')")

    return parser.parse_args()


def parse_ideal_vector(ideal_str):
    try:
        return np.array([float(x) for x in ideal_str.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Ideal vector must be a comma-separated list of floats.")


if __name__ == "__main__":
    args = parse_arguments()

    ideal = parse_ideal_vector(args.ideal_vector)

    results_path = os.path.join(args.storage_dir, args.results_file)
    with open(results_path, "rb") as f:
        results_dict = pickle.load(f)

    res_X = results_dict['decision_vecs']
    res_F = results_dict['objective_vals']
    min_joints = results_dict['min_joints']
    max_joints = results_dict['max_joints']

    if args.create_urdf:
        pyb = PybUtils(renders=args.renders)
        object_loader = LoadObjects(pyb.con)

        distances = np.linalg.norm(res_F - ideal, axis=1)
        best_idx = np.argmin(distances)
        best_decision_vector = res_X[best_idx]

        num_joints, joint_types, joint_axes, link_lengths = decode_decision_vector(best_decision_vector, min_joints, max_joints)
        best_chain = KinematicChainPyBullet(pyb.con, num_joints, joint_types, joint_axes, link_lengths, robot_name="NSGA_robot_balanced")

        best_chain.create_urdf()
        best_chain.save_urdf('best_chain_balanced')

    if args.plot_pareto:
        plot = Scatter()
        plot.add(res_F)
        plot.axis_labels = ["Pose Error", "Torque Penalty", "Joint Penalty", "Conditioning Index", 'RRMC d_joint', "RRMC Pose Error"]
        plot.show()
        del plot
        plt.close('all')
