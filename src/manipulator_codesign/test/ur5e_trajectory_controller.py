import rclpy
import numpy as np
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class UR5eTrajectoryController(Node):
    def __init__(self):
        super().__init__('ur5e_trajectory_controller')
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

    def send_trajectory(self, waypoints):
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        for i, point in enumerate(waypoints):
            traj_point = JointTrajectoryPoint()
            traj_point.positions = point
            traj_point.time_from_start = rclpy.duration.Duration(seconds=i+1).to_msg()
            trajectory_msg.points.append(traj_point)

        self.trajectory_pub.publish(trajectory_msg)
        self.get_logger().info('Trajectory sent to UR5e.')

def main(args=None):
    # rclpy.init(args=args)
    # controller = UR5eTrajectoryController()

    # # Define your waypoints as a list of joint position arrays
    # waypoints = [
    #     [0.0, -1.57, 1.57, 0.0, 1.57, 0.0],
    #     [0.5, -1.0, 1.0, 0.5, 1.0, 0.5],
    #     [1.0, -0.5, 0.5, 1.0, 0.5, 1.0]
    # ]

    # controller.send_trajectory(waypoints)

    # rclpy.spin_once(controller)
    # controller.destroy_node()
    # rclpy.shutdown()
    data = np.load("./data/voxel_paths_parallelepiped.npy")
    print(data[:5, :, 0])
    print(data.shape)

if __name__ == '__main__':
    main()
