import pickle
import pandas as pd
import numpy as np

# 1) Load your results
with open('/home/marcus/IMML/manipulator_codesign/data/nsga2_results/results_20250517_111213.pkl', 'rb') as f:
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
    by=['joint_count', 'conditioning_index'],
    ascending=[False, True]   # descending on joint_count
)

# 4) Pick top‐5
top5 = df_sorted.head(10)
top5_X = X[top5.index, :]

print("Top 5 (min pose_error, min cond_idx, max joint_count):")
print(top5)
# print("\nCorresponding decision vectors:")
# print(top5_X)