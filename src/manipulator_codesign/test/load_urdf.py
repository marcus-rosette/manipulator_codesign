import pybullet as p
import pybullet_data
import time
import numpy as np

# Initialize PyBullet simulation
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity for the simulation
p.setGravity(0, 0, -9.81)

# Load the plane URDF
plane_id = p.loadURDF("plane.urdf")

# Load the robot URDF
# robotId = p.loadURDF("/home/marcus/IMML/manipulator_codesign/urdf/ur5e/ur5e.urdf", basePosition=[0, 0, 0], useFixedBase=True)
# robotId = p.loadURDF("/home/marcus/IMML/manipulator_codesign/urdf/maybe.urdf", basePosition=[0, 0, 0], useFixedBase=True)
robotId = p.loadURDF("/home/marcus/IMML/manipulator_codesign/urdf/test_robot_6.urdf", basePosition=[0, 0, 0], useFixedBase=True)

num_joints = p.getNumJoints(robotId)
end_effector_index = num_joints - 1

controllable_idx = []
for joint in range(p.getNumJoints(robotId)):
    if p.getJointInfo(robotId, joint)[2] == p.JOINT_REVOLUTE: 
        controllable_idx.append(p.getJointInfo(robotId, joint)[0])

# print([p.getJointState(robotId, i)[0] for i in controllable_idx])

# Set initial joint positions (if needed)
initial_joint_positions = [0.5, 0, 1.57, 0, 1.57, 0]  # Replace with the initial joint positions for your robot

for i, joint_index in enumerate(controllable_idx):
    p.resetJointState(robotId, joint_index, initial_joint_positions[i])

link_state = p.getLinkState(robotId, end_effector_index)
link_position = np.array(link_state[0])


# # Simulation loop to keep the GUI open
while True:
    p.stepSimulation()
    time.sleep(1./240.)  # Simulate at 240 Hz
