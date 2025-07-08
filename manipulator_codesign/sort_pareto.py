import pickle
import pandas as pd
import numpy as np

from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_objects import LoadObjects
from manipulator_codesign.moo_decoder import decode_decision_vector
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.urdf_gen import URDFGen

# 1) Load your results
with open('/home/marcus/agrobotics/manipulator_codesign/data/nsga2_results/results_20250705_072638.pkl', 'rb') as f:
    data = pickle.load(f)
X = data['X']         # shape (n_solutions, n_vars)
F = data['F']         # shape (n_solutions, n_objs)

# 2) Build DataFrame
obj_names = ['pose_error','torque','joint_count',
             'conditioning_index','rrmc_score','rrt_path_cost']
# obj_names = ['pose_error','torque','joint_count',
#              'conditioning_index','delta_score','pos_error_rrmc', 'rrt_path_cost']
df = pd.DataFrame(F, columns=obj_names)

# 3) Sort by the three you care about
df_sorted = df.sort_values(
    by=['pose_error', 'rrt_path_cost'],
    ascending=[True, True]   # descending on joint_count
)

# 4) Pick top performers
top = df_sorted.head(25)
top_X = X[top.index, :]

print("Top 5 (min pose_error, min cond_idx, max joint_count):")
print(top)
# print("\nCorresponding decision vectors:")
# print(top5_X)


# 5) Create urdf of from top performers
pyb = PybUtils(renders=False)
object_loader = LoadObjects(pyb.con)

num_joints, joint_types, joint_axes, link_lengths = decode_decision_vector(top_X[9], 5, 7)
print(f"Number of joints: {num_joints}")
print(f"Joint types: {joint_types}")
print(f"Joint axes: {joint_axes}")
print(f"Link lengths: {link_lengths}")
joint_types = [URDFGen.map_joint_type_inverse(joint_type) for joint_type in joint_types]
best_chain = KinematicChainPyBullet(pyb.con, [0, 0, 0], num_joints, joint_types, joint_axes, link_lengths, robot_name="NSGA_robot_balanced")

best_chain.create_urdf()
best_chain.save_urdf('agbot')