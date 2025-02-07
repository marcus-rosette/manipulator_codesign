import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.spatial.transform import Rotation as R


def linear_interp_path(end_positions, start_positions, steps=100):
    """ Interpolate linear joint positions between a start and end configuration

    Args:
        end_positions (float list): end joint configuration
        start_positions (float list): start joint configuration.
        steps (int, optional): number of interpolated positions. Defaults to 100.

    Returns:
        list: interpolated path
    """
    interpolated_joint_angles = [np.linspace(start, end, steps) for start, end in zip(start_positions, end_positions)]
    return [tuple(p) for p in zip(*interpolated_joint_angles)]

def test():
    # Start the PyBullet 
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load default data
    
    robot_id = p.loadURDF("./urdf/ur5e/ur5e_cart.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

    num_joints = p.getNumJoints(robot_id)

    # Get the end-effector link index 
    end_effector_index = num_joints - 3

    controllable_joint_idx = [
        p.getJointInfo(robot_id, joint)[0]
        for joint in range(num_joints)
        if p.getJointInfo(robot_id, joint)[2] in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}
    ]

    target_ee_pos = [0, 0.8, 1.2]
    end_positions = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            target_ee_pos,
            residualThreshold=0.01
        )

    # Number of steps in the interpolation
    steps = 100

    # Interpolate joint positions
    start_pos = [-3*np.pi/4, -np.pi/3, 2*np.pi/3, 2*np.pi/3, -np.pi/2, 0]
    joint_path = linear_interp_path(end_positions, start_positions=start_pos, steps=steps)

    print(joint_path[:3])

    # Apply the interpolated positions
    for config in joint_path:
        p.setJointMotorControlArray(robot_id, controllable_joint_idx, p.POSITION_CONTROL, targetPositions=config)
        p.stepSimulation()
        time.sleep(1./100.)

    while True:
        p.stepSimulation()


if __name__ == '__main__':
    test()
