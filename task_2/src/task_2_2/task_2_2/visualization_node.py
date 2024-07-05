import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import sys

class VisualizationNode(Node):

    def __init__(self, robot_id):

        super().__init__('robot_node_' + str(robot_id))

        self.robot_id = robot_id
        self.num_agents = 6
        self.subscription_positions = self.create_subscription(Float64MultiArray, 'robot_positions', self.listener_callback_positions, 10)
        self.subscription_targets = self.create_subscription(Float64MultiArray, 'target_positions', self.listener_callback_targets, 10)
        self.subscription_barycenter = self.create_subscription(Float64MultiArray, 'barycenter_position', self.listener_callback_barycenter, 10)
        self.subscription_trajectories = self.create_subscription(Float64MultiArray, 'robot_trajectories', self.listener_callback_trajectories, 10)
        self.publisher_robots = self.create_publisher(Marker, 'robot_visualization_marker', 10)
        self.publisher_targets = self.create_publisher(MarkerArray, 'target_visualization_marker', 10)
        self.publisher_barycenter = self.create_publisher(Marker, 'barycenter_visualization_marker', 10)
        self.publisher_trajectories = self.create_publisher(MarkerArray, 'trajectory_visualization_marker', 10)
        self.positions = None
        self.targets = None
        self.barycenter = None
        self.trajectories = None

    def listener_callback_positions(self, msg):

        self.positions = np.array(msg.data).reshape(-1, 2)
        self.update_marker()

    def listener_callback_targets(self, msg):

        self.targets = np.array(msg.data).reshape(-1, 2)
        self.update_marker()

    def listener_callback_barycenter(self, msg):

        self.barycenter = np.array(msg.data)
        self.update_marker()

    def listener_callback_trajectories(self, msg):

        self.trajectories = np.array(msg.data).reshape(-1, self.num_agents, 2)
        self.update_marker()

    def update_marker(self):

        if self.positions is not None:
            self.publish_robot_marker(self.positions[self.robot_id])
        if self.targets is not None:
            self.publish_target_markers(self.targets)
        if self.barycenter is not None:
            self.publish_barycenter_marker(self.barycenter)
        if self.trajectories is not None:
            self.publish_trajectory_markers(self.trajectories)
        # if self.positions is not None and self.targets is not None:
        #     self.get_logger().info(f'Robot {self.robot_id} received positions: {self.positions[self.robot_id]} and targets: {self.targets}')

    def publish_robot_marker(self, position):

        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robots"
        marker.id = self.robot_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0  # Blue color for robots
        
        self.publisher_robots.publish(marker)
        #self.get_logger().info(f'Published robot marker for Robot {self.robot_id} at position: {position}')

    def publish_target_markers(self, targets):

        marker_array = MarkerArray()

        for i, target in enumerate(targets):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "targets"
            marker.id = i + self.num_agents  # Ensure unique ID for each target marker
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            # Create cross marker for target
            p1 = Point(x=target[0] - 0.1, y=target[1] - 0.1, z=0.0)
            p2 = Point(x=target[0] + 0.1, y=target[1] + 0.1, z=0.0)
            p3 = Point(x=target[0] - 0.1, y=target[1] + 0.1, z=0.0)
            p4 = Point(x=target[0] + 0.1, y=target[1] - 0.1, z=0.0)

            marker.points = [p1, p2, p3, p4]
            marker_array.markers.append(marker)
            #self.get_logger().info(f'Published target marker {i} at position: {target}')

        self.publisher_targets.publish(marker_array)

    def publish_barycenter_marker(self, barycenter):

        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "barycenter"
        marker.id = self.num_agents + 1  # Unique ID for barycenter marker
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = barycenter[0]
        marker.pose.position.y = barycenter[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0  # Yellow color for barycenter
        
        self.publisher_barycenter.publish(marker)
        #self.get_logger().info(f'Published barycenter marker at position: {barycenter}')

    def publish_trajectory_markers(self, trajectories):
        
        marker_array = MarkerArray()

        for i in range(self.num_agents):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "trajectories"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0  # Purple color for trajectories

            points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in trajectories[:, i, :]]
            marker.points = points
            marker_array.markers.append(marker)

        self.publisher_trajectories.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    robot_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    node = VisualizationNode(robot_id)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()