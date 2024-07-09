import pybullet as p
import pybullet_data
import numpy as np
import time


class PlanarPruner:
    def __init__(self):
        # Connect to the PyBullet physics engine
        # p.GUI opens a graphical user interface for visualization
        # p.DIRECT runs in non-graphical mode (headless)
        self.physicsClient = p.connect(p.GUI)

        # Set the path for additional URDF and other data files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set the gravity
        p.setGravity(0, 0, -10)

        # Set the camera parameters
        camera_distance = 2                      # Distance from the target
        camera_yaw = 65                          # Yaw angle in degrees
        camera_pitch = -10                       # Pitch angle in degrees
        camera_target_position = [0, 0.75, 0.75] # Target position

        # Reset the debug visualizer camera
        p.resetDebugVisualizerCamera(
            camera_distance,
            camera_yaw,
            camera_pitch,
            camera_target_position
        )

        # Load a plane URDF 
        self.planeId = p.loadURDF("plane.urdf")

        self.start_sim()

    def load_urdf(self, urdf_name, start_pos=[0, 0, 0], start_orientation=[0, 0, 0], color=None, fix_base=True, radius=None):
        orientation = p.getQuaternionFromEuler(start_orientation)

        if radius is None:
            objectId = p.loadURDF(urdf_name, start_pos, orientation, useFixedBase=fix_base)

        else:
            # Plot points as green
            objectId = p.loadURDF(urdf_name, start_pos, globalScaling=radius, useFixedBase=fix_base)
            p.changeVisualShape(objectId, -1, rgbaColor=[0, 1, 0, 1]) 

        return objectId

    def start_sim(self):
        # Define the starting position of the branches
        start_x = 0.5
        start_y = 1

        # Define the starting position of the pruning points
        self.prune_point_0_pos = [start_x, start_y, 1.55] 
        self.prune_point_1_pos = [start_x, start_y - 0.05, 1.25] 
        self.prune_point_2_pos = [start_x, start_y + 0.05, 0.55] 
        self.radius = 0.05 

        # Load the branches
        self.leader_branchId = self.load_urdf("./urdf/leader_branch.urdf", [0, start_y, 1.6/2])
        self.top_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1.5], [0, np.pi / 2, 0])
        self.mid_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 1], [0, np.pi / 2, 0])
        self.bottom_branchId = self.load_urdf("./urdf/secondary_branch.urdf", [0, start_y, 0.5], [0, np.pi / 2, 0])

        # Load the pruning points
        # self.prune_point_0 = self.load_urdf("sphere2.urdf", self.prune_point_0_pos, radius=self.radius)
        self.prune_point_1 = self.load_urdf("sphere2.urdf", self.prune_point_1_pos, radius=self.radius)
        # self.prune_point_2 = self.load_urdf("sphere2.urdf", self.prune_point_2_pos, radius=self.radius)

        # Get manipulator
        self.robotId = p.loadURDF("./urdf/three_link_manipulator.urdf", [start_x, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robotId)
        self.num_controllable_joints = self.num_joints - 4 # 3 for the static joints in the end-effector
        print(self.num_joints)
        print(self.num_controllable_joints)
        self.joint_limits = [p.getJointInfo(self.robotId, i)[8:10] for i in range(self.num_controllable_joints)]
    
    def set_joint_positions(self, joint_positions):
        for i in range(self.num_controllable_joints):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, joint_positions[i])
    
    def get_joint_positions(self):
        return [p.getJointState(self.robotId, i)[0] for i in range(self.num_controllable_joints)]
    
    def is_collision(self):
        return len(p.getContactPoints(self.robotId)) > 0
    
    def inverse_kinematics(self, position):
        joint_positions = p.calculateInverseKinematics(self.robotId, 6, position)
        return joint_positions
    

def rrt_connect(robot, start_pos, goal_pos, max_iterations=5000, step_size=0.5, collision_check_interval=1, tolerance_distance=0.1):
    start_config = robot.inverse_kinematics(start_pos)
    goal_config = robot.inverse_kinematics(goal_pos)
    
    tree_a = [start_config]
    tree_b = [goal_config]
    
    def is_valid_configuration(config):
        robot.set_joint_positions(config)
        p.stepSimulation()
        if robot.is_collision():
            return False
        return True
    
    def connect_nodes(node1, node2):
        """Try to connect two nodes by incrementally moving from node1 towards node2"""
        for _ in range(int(np.linalg.norm(np.array(node2) - np.array(node1)) / step_size)):
            new_node = np.array(node1) + step_size * (np.array(node2) - np.array(node1)) / np.linalg.norm(np.array(node2) - np.array(node1))
            new_node = new_node.tolist()
            if not is_valid_configuration(new_node):
                return None
            if np.linalg.norm(np.array(new_node) - np.array(node2)) < step_size:
                return new_node
            node1 = new_node
        return None
    
    for iteration in range(max_iterations):
        rand_config = np.random.uniform([limit[0] for limit in robot.joint_limits], [limit[1] for limit in robot.joint_limits])
        
        nearest_node_a = min(tree_a, key=lambda node: np.linalg.norm(np.array(node) - rand_config))
        new_node_a = nearest_node_a + step_size * (rand_config - nearest_node_a) / np.linalg.norm(rand_config - nearest_node_a)
        
        if is_valid_configuration(new_node_a):
            tree_a.append(new_node_a)
            if np.linalg.norm(np.array(new_node_a) - np.array(goal_config)) < step_size:
                path = connect_nodes(new_node_a, goal_config)
                if path:
                    # Check final position against goal position with tolerance
                    final_position = robot.get_joint_positions()
                    final_distance = np.linalg.norm(np.array(final_position) - np.array(goal_config))
                    if final_distance <= tolerance_distance:
                        print(f"Goal reached within tolerance after {iteration} iterations!")
                        return tree_a + path[::-1]
                    else:
                        print(f"Goal reached, but final distance {final_distance:.3f} exceeds tolerance.")
                        continue
                else:
                    print(f"Failed to connect to the goal.")
                    continue
        
        if iteration % collision_check_interval == 0:
            print(f"Iteration: {iteration} - Still searching...")
        
        nearest_node_b = min(tree_b, key=lambda node: np.linalg.norm(np.array(node) - new_node_a))
        new_node_b = nearest_node_b + step_size * (new_node_a - nearest_node_b) / np.linalg.norm(new_node_a - nearest_node_b)
        
        if is_valid_configuration(new_node_b):
            tree_b.append(new_node_b)
            if np.linalg.norm(np.array(new_node_b) - np.array(start_config)) < step_size:
                path = connect_nodes(start_config, new_node_b)
                if path:
                    # Check final position against goal position with tolerance
                    final_position = robot.get_joint_positions()
                    final_distance = np.linalg.norm(np.array(final_position) - np.array(goal_config))
                    if final_distance <= tolerance_distance:
                        print(f"Goal reached within tolerance after {iteration} iterations!")
                        return path + tree_b[::-1]
                    else:
                        print(f"Goal reached, but final distance {final_distance:.3f} exceeds tolerance.")
                        continue
                else:
                    print(f"Failed to connect to the start.")
                    continue
    
    print("Max iterations reached, no path found within tolerance!")
    return None

def main():
    planar_pruner = PlanarPruner()
    
    start_pos = [0.0, 0.5, -0.5]
    goal_pos = planar_pruner.prune_point_1_pos
    
    path = rrt_connect(planar_pruner, start_pos, goal_pos, max_iterations=5000, tolerance_distance=1)
    
    if path is None:
        print("No path found!")
    else:
        print("Path found! Executing path...")
        for config in path:
            planar_pruner.set_joint_positions(config)
            p.stepSimulation()
            time.sleep(0.1)
        print("Path execution complete. Robot should stay at the goal position.")
        
        # Maintain the final position
        final_position = path[-1]
        while True:
            planar_pruner.set_joint_positions(final_position)
            p.stepSimulation()
            time.sleep(0.1)

    p.disconnect()


if __name__ == "__main__":
    main()
