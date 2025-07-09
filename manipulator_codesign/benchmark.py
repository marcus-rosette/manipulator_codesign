import os
import argparse
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_objects import LoadObjects
from pybullet_robokit.load_robot import LoadRobot
from pybullet_robokit.motion_planners import KinematicChainMotionPlanner
from manipulator_codesign.pose_generation import generate_northern_hemisphere_orientations, downsample_quaternions_facing_robot, sample_collision_free_poses
from manipulator_codesign.training_env import load_plant_env
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
from manipulator_codesign.urdf_to_decision_vector import urdf_to_decision_vector
from manipulator_codesign.urdf_gen import URDFGen


def get_urdf_path(user_input, default_dir, default_file):
    """
    Determines the correct URDF file path.
    
    Args:
        user_input (str): The user-provided URDF path or filename.
        default_dir (str): The default directory where URDF files are stored.
        default_file (str): The default URDF filename.
    
    Returns:
        str: The resolved URDF file path.
    """
    if os.path.isabs(user_input) and os.path.isfile(user_input):
        return user_input
    
    potential_path = os.path.join(default_dir, user_input)
    if os.path.isfile(potential_path):
        return potential_path
    
    return os.path.join(default_dir, default_file)


class BenchMarkRobot:
    def __init__(self, robot_urdf_path: str, robot_home_pos, ik_tol=0.01, renders=True, ee_link_name='ee_link', xarm=False):
        """
        Initialize the ViewRobot class.

        Args:
            robot_urdf_path (str): Path to the URDF file of the robot.
            robot_home_pos (list): Home position of the robot joints.
            ik_tol (float, optional): Tolerance for inverse kinematics. Defaults to 0.01.
            renders (bool, optional): Whether to visualize the robot in the PyBullet GUI. Defaults to True.
        """
        self.pyb = PybUtils(renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)
        
        self.xarm = xarm

        # Load a new environment with plant objects
        self.object_loader.collision_objects.extend(load_plant_env(self.pyb.con))

        if self.xarm:
            # Box size from URDF: full extents, but PyBullet needs half extents
            full_extents = [0.75, 0.25, 0.25]
            half_extents = [dim / 2.0 for dim in full_extents]

            # Create collision and visual shapes
            collision_shape = self.pyb.con.createCollisionShape(self.pyb.con.GEOM_BOX, halfExtents=half_extents)
            visual_shape = self.pyb.con.createVisualShape(
                shapeType=self.pyb.con.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.5, 0.5, 0.5, 1.0],  # gray with full opacity
            )

            # Create the body with mass and inertia (as specified)
            mass = 1.0
            base_position = [0, 0, 0.125]
            base_orientation = [0, 0, 0, 1]  # quaternion

            body_id = self.pyb.con.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=base_position,
                baseOrientation=base_orientation,
            )

            self.object_loader.collision_objects.append(body_id)

        self.robot_urdf_path = robot_urdf_path

        self.ik_tol = ik_tol
        self.ee_link_name = ee_link_name

    def load_optimized_manip(self, robot_base_position, target_points):
        """
        Load an optimized manipulator from a URDF file.

        Args:
            urdf_path (str): Path to the URDF file.
            robot_home_pos (list, optional): Home position of the robot joints. Defaults to None.
            ee_link_name (str, optional): Name of the end-effector link. Defaults to 'ee_link'.
        """
        # decode and build kinematic chain
        joint_count, joint_types, joint_axes, link_lengths = urdf_to_decision_vector(self.robot_urdf_path, self.ee_link_name)

        joint_types = [URDFGen.map_joint_type_inverse(jt) for jt in joint_types]
        joint_axes = [(0.0, 0.0, 0.0) if isinstance(item, list) and all(isinstance(subitem, tuple) for subitem in item) else item
                    for item in joint_axes]
        joint_axes = [' '.join(map(str, map(int, ja))) for ja in joint_axes]
        joint_axes = [URDFGen.map_axis_inverse(ja) for ja in joint_axes]

        ch = KinematicChainPyBullet(
            self.pyb.con, robot_base_position,
            joint_count, joint_types, joint_axes, link_lengths,
            collision_objects=self.object_loader.collision_objects,
            ee_link_name=self.ee_link_name,
        )
        if not ch.is_built and not self.xarm:
            ch.build_robot()
        elif self.xarm:
            ch.urdf_path = self.robot_urdf_path
        ch.load_robot()

        # Sample collision-free poses (target points with orientations)
        candidate_orientations = generate_northern_hemisphere_orientations(num_orientations=30)
        N = len(target_points)
        M = len(candidate_orientations)
        candidate_poses = np.empty((N, M, 2), dtype=object)
        for i, target_pt in enumerate(target_points):
            # Generate poses for each target point with all candidate orientations
            candidate_poses[i, :, :] = [(target_pt, quat) for quat in candidate_orientations]
        target_poses, _ = ch.sample_collision_free_poses(candidate_poses)

        ch.compute_chain_metrics(target_poses, None)

        return {
            'pose_error':                 ch.mean_pose_error,
            'pose_error_std':            ch.std_pose_error,

            'rrt_path_cost':             ch.mean_rrt_path_cost,
            'rrt_path_cost_std':         ch.std_rrt_path_cost,

            'torque':                    ch.mean_torque,
            'torque_std':                ch.std_torque,

            'joint_count':               ch.num_joints,
            'conditioning_index':        ch.global_conditioning_index,

            'delta_joint_score_rrmc':    ch.mean_delta_joint_score_rrmc,
            'delta_joint_score_rrmc_std': ch.std_delta_joint_score_rrmc,

            'pos_error_rrmc':            ch.mean_pos_error_rrmc,
            'pos_error_rrmc_std':        ch.std_pos_error_rrmc
        }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a robot in PyBullet simulation.")
    parser.add_argument("-u", "--urdf_path", type=str, default="example_6dof_manipulator.urdf", 
                        help="URDF file path or name (default: example_6dof_manipulator.urdf)")
    parser.add_argument("--render", action="store_true", help="Enable rendering in PyBullet")
    parser.add_argument("--no-render", action="store_false", dest="render", help="Disable rendering in PyBullet")
    
    parser.set_defaults(render=True)  # Default to True
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_urdf_dir = os.path.join(script_dir, 'urdf', 'robots')
    default_urdf_file = os.path.join(default_urdf_dir, "example_6dof_manipulator.urdf")
    
    robot_urdf_path = get_urdf_path(args.urdf_path, default_urdf_dir, default_urdf_file)

    if "agbot" in robot_urdf_path or "test_robot" in robot_urdf_path:
        ee_link_name = 'end_effector'
        xarm = False
    else:
        ee_link_name = 'link7'
        xarm = True
    
    robot_home_pos = None

    print(f"\nLoading robot from: {robot_urdf_path}\n")
    
    bench_mark = BenchMarkRobot(robot_urdf_path=robot_urdf_path, 
                           renders=args.render,
                           robot_home_pos=robot_home_pos,
                           ik_tol=0.01,
                           ee_link_name=ee_link_name,
                           xarm=xarm)
    
    target_points = np.array([
            [-0.20, -0.50, 0.28],
            [-0.12, -0.57, 0.28],
            [0.12, -0.35, 0.28],
            [0.1, -0.2, 0.28],
            [0.18, -0.7, 0.28],
            [0.25, -0.15, 0.28],
            [0.48, -0.35, 0.28],

            [0.20, 0.50, 0.28],
            [0.12, 0.57, 0.28],
            [-0.12, 0.35, 0.28],
            [-0.1, 0.2, 0.28],
            [-0.18, 0.7, 0.28],
            [-0.25, 0.15, 0.28],
            [-0.48, 0.35, 0.28],
        ])
    
    results = bench_mark.load_optimized_manip(
                robot_base_position=[0, 0, 0.25],
                target_points=target_points,
                )
    
    print("\nBenchmark Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")    