import numpy as np
from scipy.linalg import svdvals
from manipulator_codesign.pyb_utils import PybUtils
from manipulator_codesign.load_objects import LoadObjects
from manipulator_codesign.kinematic_chain import KinematicChainPyBullet
import pytest

# Define your test parameters once
TEST_ANGLES = [
    (0.0, 0.0), 
    (np.pi/4, np.pi/6),
    (np.pi/3, -np.pi/4),
    (-np.pi/2, np.pi/3),
    (np.pi/6, -np.pi/3)
]

def compute_analytical_jacobian(l1, l2, theta1, theta2):
    """
    Analytical Jacobian for a 2-link planar arm rotating in the YZ plane,
    with joints rotating about the x axis and links initially aligned along z.
    This formulation is corrected to match the PyBullet coordinate frame.
    
    The end-effector position is defined as:
        p = [0,
             -l1*sin(theta1) - l2*sin(theta1+theta2),
             l1*cos(theta1) + l2*cos(theta1+theta2)]
    
    Args:
        l1 (float): Length of the first link.
        l2 (float): Length of the second link.
        theta1 (float): Angle of the first joint in radians.
        theta2 (float): Angle of the second joint in radians.
        
    Returns:
        np.ndarray: The 6x2 Jacobian matrix.
    """
    theta12 = theta1 + theta2

    # Compute partial derivatives for the position in the YZ plane.
    dy_dtheta1 = -l1 * np.cos(theta1) - l2 * np.cos(theta12)
    dz_dtheta1 = -l1 * np.sin(theta1) - l2 * np.sin(theta12)
    dy_dtheta2 = -l2 * np.cos(theta12)
    dz_dtheta2 = -l2 * np.sin(theta12)

    # Construct the 6x2 Jacobian.
    # Note: The translational part is:
    #   [0, 0] for the x component (no motion in x),
    #   [dy_dtheta1, dy_dtheta2] for the y component,
    #   [dz_dtheta1, dz_dtheta2] for the z component.
    # The angular part is [1,1] for rotation about x and zeros for y and z.
    J = np.array([
        [0.0,        0.0],          # linear x
        [dy_dtheta1, dy_dtheta2],   # linear y
        [dz_dtheta1, dz_dtheta2],   # linear z
        [1.0,        1.0],          # angular x
        [0.0,        0.0],          # angular y
        [0.0,        0.0]           # angular z
    ])

    return J

import numpy as np

def compute_analytical_gravity_torques(l1, l2, theta1, theta2, m1=1.0, m2=1.0, g=9.81):
    """
    Computes the joint torques due to gravity for a 2-link planar arm.
    
    The arm is assumed to be in the YZ plane with joints rotating about the x-axis.
    The end-effector (and link 2) effective length is increased by 0.1, so that:
    
        p = [0,
             -l1*sin(theta1) - (l2+0.1)*sin(theta1+theta2),
             l1*cos(theta1) + (l2+0.1)*cos(theta1+theta2)]
    
    Each link's center of mass is assumed to be at half its effective length.
    For link 1: effective length = l1.
    For link 2: effective length = l2 + 0.1.
    
    Potential energies:
      U1 = m1*g*(l1/2 * cos(theta1))
      U2 = m2*g*(l1*cos(theta1) + (l2+0.1)/2 * cos(theta1+theta2))
    
    Joint torques (gravity compensation) are given by:
      tau_i = -dU/dtheta_i
    
    Args:
        l1 (float): Length of the first link.
        l2 (float): Nominal length of the second link.
        theta1 (float): Angle of the first joint in radians.
        theta2 (float): Angle of the second joint in radians.
        m1 (float): Mass of the first link.
        m2 (float): Mass of the second link.
        g (float): Gravitational acceleration.
        
    Returns:
        np.ndarray: Joint torques [tau1, tau2].
    """
    # Derivative of potential energy with respect to theta1.
    dU1_dtheta1 = m1 * g * (-l1/2 * np.sin(theta1))
    dU2_dtheta1 = m2 * g * (-l1 * np.sin(theta1) - (l2/2) * np.sin(theta1+theta2))
    tau1 = -(dU1_dtheta1 + dU2_dtheta1)
    
    # Derivative of potential energy with respect to theta2.
    dU2_dtheta2 = m2 * g * (-(l2/2) * np.sin(theta1+theta2))
    tau2 = -dU2_dtheta2

    # Flip the overall sign to match PyBullet's torque convention.
    return -np.array([tau1, tau2])

def compute_analytical_singular_values(l1, l2, theta1, theta2):
    """
    Compute the singular values of the Jacobian analytically.
    
    This derivation accounts for both the translational and rotational components
    of the Jacobian. The translational part yields:
      a_t = l1^2 + l2^2 + 2*l1*l2*cos(theta2)
      b_t = l1*l2*cos(theta2) + l2^2
      c_t = l2^2
    and the rotational part contributes a constant 1 to the diagonal and off-diagonal
    terms. Thus, the full J^T J matrix has:
      A = a_t + 1,  B = b_t + 1,  C = c_t + 1.
    
    The eigenvalues of J^T J are then:
      lambda_{1,2} = (A + C ± sqrt((A - C)^2 + 4*B^2)) / 2.
    
    Args:
        l1 (float): Length of the first link.
        l2 (float): Length of the second link.
        theta1 (float): Angle of the first joint (unused in this derivation).
        theta2 (float): Angle of the second joint.
        
    Returns:
        tuple: (sigma_max, sigma_min) where sigma_max >= sigma_min.
    """
    # Contributions from the translational part plus the rotational part (+1)
    A = l1**2 + l2**2 + 2 * l1 * l2 * np.cos(theta2) + 1
    B = l1 * l2 * np.cos(theta2) + l2**2 + 1
    C = l2**2 + 1

    # Compute eigenvalues of the 2x2 matrix
    temp = np.sqrt((A - C)**2 + 4 * B**2)
    lambda1 = (A + C + temp) / 2
    lambda2 = (A + C - temp) / 2

    sigma1 = np.sqrt(lambda1)
    sigma2 = np.sqrt(lambda2)
    
    return (max(sigma1, sigma2), min(sigma1, sigma2))

def compute_pyb_config_params(l1, l2, theta1, theta2):
    """
    Compute the Jacobian matrix for a 2-DOF planar robot using PyBullet.
    This function builds a 2-link planar robotic arm, sets its joint positions,
    and calculates the Jacobian matrix at the specified joint angles.
    Args:
        l1 (float): Length of the first link.
        l2 (float): Length of the second link.
        theta1 (float): Joint angle of the first link in radians.
        theta2 (float): Joint angle of the second link in radians.
    Returns:
        np.ndarray: The Jacobian matrix of the robot at the specified joint angles.
    """
    pyb_utils = PybUtils()
    _ = LoadObjects(pyb_utils.con)
    
    # Create a simple 2-link planar arm
    robot = KinematicChainPyBullet(pyb_utils.con, 2, [1, 1], ['x', 'x'], [l1, l2], 'link1')

    robot.build_robot()
    robot.load_robot()
    robot.robot.reset_joint_positions([theta1, theta2])

    return robot.robot.get_jacobian([theta1, theta2], local_pos=[0, 0, l2]), robot.robot.inverse_dynamics([theta1, theta2])

@pytest.mark.parametrize("theta1, theta2", TEST_ANGLES)
def test_jacobian(theta1, theta2):
    l1 = 1.0
    l2 = 0.5
    
    # Compute the Jacobian using PyBullet
    J_pyb, _ = compute_pyb_config_params(l1, l2, theta1, theta2)

    # Compute the Jacobian analytically
    J_analytical = compute_analytical_jacobian(l1, l2, theta1, theta2)
    
    np.testing.assert_almost_equal(J_pyb, J_analytical, decimal=6,
                                   err_msg=f"Jacobian mismatch for theta1={theta1}, theta2={theta2}")

@pytest.mark.parametrize("theta1, theta2", TEST_ANGLES)
def test_jacobian_singular_values(theta1, theta2):
    l1 = 1.0
    l2 = 0.5
    
    J = compute_analytical_jacobian(l1, l2, theta1, theta2)
    svd_vals = svdvals(J)
    sigma_num_max = max(svd_vals)
    sigma_num_min = min(svd_vals)
    
    sigma_analytical_max, sigma_analytical_min = compute_analytical_singular_values(l1, l2, theta1, theta2)
    
    np.testing.assert_almost_equal(sigma_num_max, sigma_analytical_max, decimal=6,
                                   err_msg=f"Mismatch in sigma_max for theta1={theta1}, theta2={theta2}")
    np.testing.assert_almost_equal(sigma_num_min, sigma_analytical_min, decimal=6,
                                   err_msg=f"Mismatch in sigma_min for theta1={theta1}, theta2={theta2}")
    
@pytest.mark.parametrize("theta1, theta2", TEST_ANGLES)
def test_gravity_torque_com(theta1, theta2):
    l1 = 1.0
    l2 = 0.5
    
    _, tau_pyb = compute_pyb_config_params(l1, l2, theta1, theta2)
    tau_analytical = compute_analytical_gravity_torques(l1, l2, theta1, theta2)

    print(f"theta1={theta1}, theta2={theta2}")
    print(f"PyBullet torque: {tau_pyb}")
    print(f"Analytical torque: {tau_analytical}")
    print(f"Torque difference: {np.array(tau_pyb) - tau_analytical}")
    
    np.testing.assert_almost_equal(tau_pyb, tau_analytical, decimal=6,
                                   err_msg=f"Joint torque mismatch for theta1={theta1}, theta2={theta2}")
