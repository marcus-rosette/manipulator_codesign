import roboticstoolbox as rtb
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R


def ik_solver(robot, target_position, target_quaternion):
    # Convert quaternion to rotation matrix
    r = R.from_quat(target_quaternion)
    rotation_matrix = r.as_matrix()

    # Create the target pose
    T = SE3(target_position) * SE3(rotation_matrix)

    # Solve the inverse kinematics
    q = robot.ikine_LM(T)  # Use the Levenberg-Marquardt method for IK
    return q

def orientation_error(robot, q, target_quaternion):
    # Compute the end-effector pose for the given joint configuration
    T_current = robot.fkine(q)
    current_quaternion = R.from_matrix(T_current.R).as_quat()

    # Compute the orientation error
    r_target = R.from_quat(target_quaternion)
    r_current = R.from_quat(current_quaternion)
    error_rotation = r_target.inv() * r_current
    error_angle = error_rotation.magnitude()  # Error angle in radians
    
    return error_angle


if __name__ == '__main__':
    robot = rtb.models.DH.UR5()

    target_pos = [0, 0.5, 0.5]
    target_ori = [0.5, 0.5, 0, 0.5]

    q = ik_solver(robot, target_pos, target_ori)