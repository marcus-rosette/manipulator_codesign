# IMML Optimal Manipulator Co-Design

## PyBullet Manipulability Path Simulation

This package provides a PyBullet-based simulation framework for generating and caching paths to high-manipulability configurations for robotic arms. It is designed to integrate with UR robots (or other manipulators with valid URDF) and supports path planning, inverse kinematics (IK) computation, and manipulability analysis.

### Features

- **Inverse Kinematics Solutions**: Compute IK solutions for end-effector approach positions and orientations sampled on a hemisphere surrounding the target point.
- **Path Caching**: Cache robot joint trajectories to manipulability-optimized configurations.
- **Collision Checking**: Avoid ground collisions and ensure safe trajectories.
- **Manipulability Analysis**: Evaluate and optimize configurations based on manipulability metrics.
- **Workspace Sampling**: Workspace voxelization is used to discretize and sample regions that are reachable by the robot.
- **Simulation Visualization**: Visualize the robot and simulation environment with PyBullet.