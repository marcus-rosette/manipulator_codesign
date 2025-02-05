import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R

# Connect to PyBullet and load the UR5e robot URDF
# Start the PyBullet simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load default data

robot_id = p.loadURDF("./urdf/ur5e/ur5e_cutter_cart.urdf", useFixedBase=True)

num_joints = p.getNumJoints(robot_id)

# Get the end-effector link index for UR5e (assuming it's the 6th joint)
end_effector_index = num_joints - 2  # Change if your URDF differs

controllable_joint_idx = []
for joint in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint)
    joint_type = joint_info[2]

    if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC: 
        controllable_joint_idx.append(p.getJointInfo(robot_id, joint)[0])

# Target position and orientation for the end-effector
target_position = [0.5, 0.5, 1.25]  # Random point in space
look_at_point = [0, 1, 1]    # Point where the end-effector should face

# Visualize the look_at_point as a sphere
look_at_sphere = p.loadURDF("sphere2.urdf", look_at_point, globalScaling=0.05, useFixedBase=True)

# Calculate the direction vector from the end-effector to the look-at point
direction_vector = np.array(look_at_point) - np.array(target_position)
direction_vector /= np.linalg.norm(direction_vector)  # Normalize

# Create a rotation matrix from the direction vector
up_vector = np.array([0, 0, 1])  # Assuming the z-axis is the up direction
right_vector = np.cross(up_vector, direction_vector)
right_vector /= np.linalg.norm(right_vector)  # Normalize
up_vector = np.cross(direction_vector, right_vector)  # Recompute up vector

rotation_matrix = np.column_stack((right_vector, up_vector, direction_vector))  # [R_x, R_y, R_z]

# Convert the rotation matrix to a quaternion using SciPy
rotation = R.from_matrix(rotation_matrix)
orn_quat = rotation.as_quat()  # Quaternion as (x, y, z, w)

# Use PyBullet's inverse kinematics to compute joint angles
joint_angles = p.calculateInverseKinematics(
    robot_id,
    end_effector_index,
    target_position,
    orn_quat
)

# Iterate over the joints and set their positions
for i, joint_idx in enumerate(controllable_joint_idx):
    p.setJointMotorControl2(
        bodyIndex=robot_id,
        jointIndex=joint_idx,
        controlMode=p.POSITION_CONTROL,
        targetPosition=joint_angles[i]
    )

# Run the simulation to observe the result
while True:
    p.stepSimulation()

p.disconnect()
