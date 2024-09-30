# Blind-Source-Separation-using-ML-methods
This is an open project of BSS using ML algorithms. Part of this project uses code from this repository: Shirazi, S.Y. 2022: hdEMG-Decompostion (github.com/neuromechanist/hdEMG-Decomposition/tag/0.1), GitHub. https://doi.org/10.5281/zenodo.7106379

<build_depend>rclpy</build_depend>
  <exec_depend>rclpy</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>

'talker_node = talker.talker_node:main',

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
class LocalizationPublisher(Node):
    def __init__(self):
        super().__init__('localization_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'aruco_pose', 10)
        self.timer = self.create_timer(1, self.timer_callback)
    def timer_callback(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_link'
        # Example pose data -- replace with your localization data
        pose_msg.pose.position.x = 1.0
        pose_msg.pose.position.y = 2.0
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        self.publisher_.publish(pose_msg)
        self.get_logger().info(f'Publishing: {pose_msg.pose}')
def main(args=None):
    rclpy.init(args=args)
    localization_publisher = LocalizationPublisher()
    rclpy.spin(localization_publisher)
    localization_publisher.destroy_node()
    rclpy.shutdown()
