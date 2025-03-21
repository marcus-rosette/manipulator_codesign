# IMML Optimal Manipulator Co-Design

## PyBullet Manipulability Path Simulation

## Overview
This repository provides a modular and flexible software framework for designing and optimizing robotic manipulators. The framework integrates **URDF generation**, **physics-based simulation**, and **genetic algorithms** to automate the design and evaluation of manipulators for agricultural and other task-specific applications. By leveraging simulation-based optimization, this tool generates manipulator configurations that maximize **workspace coverage**, enhance **dexterity**, and minimize **joint torques**.

## Features
- **Automated URDF Generation**: Uses `URDFGen` to define custom manipulator structures.
- **Physics-Based Simulation**: Leverages PyBullet for kinematics, dynamics, and collision evaluation.
- **Genetic Algorithm Optimization**: Evolves manipulator designs based on performance metrics.
- **Task-Oriented Evaluation**: Assesses manipulators based on inverse kinematics, motion planning, and workspace efficiency.
- **Simulation Visualization**: Provides tools to view and analyze optimized designs.

## Installation
Ensure Python 3.8+ is installed, then install dependencies and package itself:

```bash
python -m pip install .
```

## Usage

### Running the Optimization
To execute the genetic algorithm with default parameters, run:

```bash
python manipulator_codesign/examples/run_genetic_algorithm.py
```

To customize optimization settings, specify parameters such as population size, number of generations, mutation rate, and crossover rate:
```bash
python run_genetic_algorithm.py -p 200 -g 100 -m 0.3 -c 0.7
```
Where:
- -p (--population_size): Number of candidate solutions in each generation (default: 100).
- -g (--generations): Number of generations the algorithm will run (default: 50).
- -m (--mutation_rate): Probability of mutation in genetic variation (default: 0.3).
- -c (--crossover_rate): Probability of crossover between parent solutions (default: 0.7).

### Visualizing the Optimized Manipulator
Once the optimization process completes, you can load and test the generated URDF in the PyBullet environment:

```bash
python manipulator_codesign/view_robot.py -u path/to/robot.urdf
```

If no URDF path is specified, the script will attempt to load a default model from the urdf/robots/ directory.

The visualization script:

- Loads the generated URDF.
- Generates random target end-effector poses.
- Performs inverse kinematics to evaluate motion feasibility.
- Allows user interaction via Enter key to step through different poses.

This process helps verify that the optimized manipulator is capable of executing the desired movements before real-world implementation.
