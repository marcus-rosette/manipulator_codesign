import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

data = np.loadtxt("./data/ur5_planar_arc_manip.csv")

point = data[:, :3]
orientation = data[:, 3:7]
manipulability = data[:, 7:]

num_points = len(manipulability)

prune_point = np.array([0.5, 0.95, 1.1])
branch_loc = np.array([1.0, 1.0])

# Normalize values for color mapping
norm = mcolors.Normalize(vmin=np.min(manipulability), vmax=np.max(manipulability))
cmap = plt.get_cmap('copper')

# Plot the arc as a heatmap
plt.figure(figsize=(8, 8))
for i in range(num_points):
    plt.scatter(point[i, 1], point[i, 2], color=cmap(norm(manipulability[i])), s=100)

plt.plot(prune_point[1], prune_point[2], marker='o', color='limegreen', markersize=15)
plt.scatter(branch_loc[0], branch_loc[1], s=5000, edgecolor='forestgreen', facecolors='none', linewidths=3)

# Add custom legends (mainly to reduce the marker size)
prune_marker = plt.Line2D([0], [0], marker='o', color='limegreen', markersize=15, linestyle='None', label='Prune Point')
branch_marker = plt.Line2D([0], [0], marker='o', color='forestgreen', markersize=15, linestyle='None', markeredgewidth=2, markerfacecolor='none', label='Branch Location')
plt.legend(handles=[prune_marker, branch_marker], markerscale=0.5)

# Add a color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Manipulability')

plt.title('Arc Heatmap')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0.83, 1.025) 
plt.ylim(0.98, 1.157)
plt.tight_layout()
plt.show()
