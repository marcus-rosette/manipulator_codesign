import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from urdf_gen import URDFGen
from load_robot import LoadRobot

# Base class with shared parameters and methods
class KinematicChainBase:
    def __init__(self, num_joints, joint_types, joint_axes, link_lengths, 
                 robot_name='silly_robot', save_urdf_dir=None,
                 joint_limit_prismatic=(-0.5, 0.5), joint_limit_revolute=(-np.pi, np.pi)):
        """
        Initialize the kinematic chain for the robot.

        Args:
            num_joints (int): Number of joints in the kinematic chain.
            joint_types (list of int): Types of joints (0 for prismatic, 1 for revolute).
            joint_axes (list of list of float): Axes of each joint.
            link_lengths (list of float): Lengths of each link in the kinematic chain.
            robot_name (str, optional): Name of the robot. Defaults to 'silly_robot'.
            save_urdf_dir (str, optional): Directory to save the URDF file. Defaults to None.
            joint_limit_prismatic (tuple of float, optional): Joint limits for prismatic joints. Defaults to (-0.5, 0.5).
            joint_limit_revolute (tuple of float, optional): Joint limits for revolute joints. Defaults to (-np.pi, np.pi).
        """
        self.num_joints = num_joints
        self.joint_types = joint_types
        self.joint_axes = joint_axes
        self.link_lengths = link_lengths
        self.robot_name = robot_name
        self.joint_limit_prismatic = joint_limit_prismatic
        self.joint_limit_revolute = joint_limit_revolute
        self.joint_limits = [
            self.joint_limit_prismatic if jt == 0 else self.joint_limit_revolute 
            for jt in joint_types
        ]

        self.save_urdf_dir = save_urdf_dir
        self.urdf_gen = URDFGen(self.robot_name, self.save_urdf_dir)

    def create_urdf(self):
        """
        Generates a URDF (Unified Robot Description Format) representation of the manipulator.

        This method uses the URDF generator to create a URDF file for the manipulator based on
        the provided joint axes, joint types, link lengths, and joint limits.
        """
        self.urdf_gen.create_manipulator(self.joint_axes, self.joint_types, self.link_lengths, self.joint_limits)
    
    def save_urdf(self, filename):
        """
        Save the URDF (Unified Robot Description Format) file.

        Args:
            filename (str): The name of the file where the URDF will be saved.
        """
        self.urdf_gen.save_urdf(filename)

    def describe(self):
        """
        Prints a description of the kinematic chain, including the number of joints,
        joint types, joint axes, and link lengths.
        """
        print("\nKinematic Chain Description:")
        print(f"  Number of Joints: {self.num_joints}")
        print(f"  Joint Types: {self.joint_types}")
        print(f"  Joint Axes: {self.joint_axes}")
        print(f"  Link Lengths: {self.link_lengths}")

    def compute_fitness(self, target):
        """
        Compute the error between the chain’s end-effector and a target position.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")


# --- Robotics Toolbox Implementation ---
class KinematicChainRTB(KinematicChainBase):
    def __init__(self, num_joints, joint_types, joint_axes, link_lengths, **kwargs):
        super().__init__(num_joints, joint_types, joint_axes, link_lengths, **kwargs)
        # Build the robot using the DH convention (RTB)
        links = []
        for i in range(self.num_joints):
            if self.joint_types[i] == 1:  # Revolute joint
                links.append(rtb.RevoluteDH(
                    a=self.link_lengths[i] if self.joint_axes[i] in ['x', 'y'] else 0,
                    d=self.link_lengths[i] if self.joint_axes[i] == 'z' else 0,
                    alpha=0, offset=0,
                    qlim=self.joint_limits[i]
                ))
            else:  # Prismatic joint
                links.append(rtb.PrismaticDH(
                    a=self.link_lengths[i] if self.joint_axes[i] in ['x', 'y'] else 0,
                    theta=0, alpha=0, offset=0,
                    qlim=(0, self.link_lengths[i])
                ))
        self.robot = rtb.DHRobot(links, name=self.robot_name)

    def compute_fitness(self, target):
        # Use RTB’s inverse kinematics method.
        ik_solution = self.robot.ikine_LM(SE3(target), tol=0.01, slimit=100, ilimit=10, joint_limits=True)
        return ik_solution.residual  # Lower is better


# --- PyBullet Implementation ---
class KinematicChainPyBullet(KinematicChainBase):
    def __init__(self, pyb_con, num_joints, joint_types, joint_axes, link_lengths, **kwargs):
        """
        pyb_con: A connection object from your PyBullet utilities.
        """
        super().__init__(num_joints, joint_types, joint_axes, link_lengths, **kwargs)
        self.pyb_con = pyb_con
        
        # Create a URDF for this chain.
        self.create_urdf()
        # Save a temporary URDF file to load the robot.
        urdf_path = self.urdf_gen.save_temp_urdf()
        
        # Load the robot into PyBullet.
        self.robot = LoadRobot(self.pyb_con, 
                               urdf_path, 
                               start_pos=[0, 0, 0], 
                               start_orientation=self.pyb_con.getQuaternionFromEuler([0, 0, 0]),
                               home_config=[0] * self.num_joints)

    def compute_fitness(self, target):
        # Compute fitness by solving IK in PyBullet.
        joint_config = self.robot.inverse_kinematics(target, pos_tol=0.1)
        self.robot.reset_joint_positions(joint_config)
        ee_pos, _ = self.robot.get_link_state(self.robot.end_effector_index)
        error = np.linalg.norm(np.array(target) - np.array(ee_pos))
        return error
