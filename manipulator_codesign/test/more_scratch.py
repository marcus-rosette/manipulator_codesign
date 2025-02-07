import networkx as nx
import numpy as np
from networkx.algorithms import approximation

# 3D points
points = np.array([[1, 2, 3], [1, 3, 4], [3, 3, 5], [5, 4, 6]])

# Create a graph and add nodes
G = nx.complete_graph(len(points))

# Assign Euclidean distances as edge weights
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        dist = np.linalg.norm(points[i] - points[j])
        G[i][j]['weight'] = dist

# Solve the TSP
tsp_solution = approximation.traveling_salesman_problem(G, weight='weight')
print("TSP path:", tsp_solution)


for i, idx in enumerate(tsp_solution):
    print(f'Points {i}: {points[idx]}')

sorted_coords = np.array([points[idx] for idx in tsp_solution])

for i in range(len(sorted_coords)):
    if i == 0:
        # Print the first coordinate
        print(sorted_coords[i])
    elif i == len(sorted_coords) - 1:
        # Print the last coordinate
        print(sorted_coords[i])
    else:
        # Print the adjacent pairs
        print(sorted_coords[i - 1], sorted_coords[i])

print(points[-1])