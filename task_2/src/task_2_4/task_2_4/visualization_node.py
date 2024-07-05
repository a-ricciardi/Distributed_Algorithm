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
        self.subscription_positions = self.create_subscription(Float64MultiArray, 'robot_positions', self.listener_callback_positions, 10)
        self.subscription_targets = self.create_subscription(Float64MultiArray, 'target_positions', self.listener_callback_targets, 10)
        self.subscription_barycenter = self.create_subscription(Float64MultiArray, 'barycenter_position', self.listener_callback_barycenter, 10)
        self.subscription_trajectories = self.create_subscription(Float64MultiArray, 'robot_trajectories', self.listener_callback_trajectories, 10)
        self.publisher_robots = self.create_publisher(Marker, 'robot_visualization_marker', 10)
        self.publisher_targets = self.create_publisher(MarkerArray, 'target_visualization_marker', 10)
        self.publisher_barycenter = self.create_publisher(Marker, 'barycenter_visualization_marker', 10)
        self.publisher_boundaries = self.create_publisher(MarkerArray, 'boundary_visualization_marker', 10)
        self.num_agents = 5
        self.positions = None
        self.targets = None
        self.barycenter = None
        self.trajectories = None

        # Boundary parameters
        self.boundary_a = 0.3
        self.boundary_b = -3.0
        self.boundary_c = 0.5
        self.x_range = np.linspace(0, 20, 100)

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
        self.publish_boundary_markers()

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
            marker.id = i + self.robot_id  # Unique ID for each target marker
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

        self.publisher_boundaries.publish(marker_array)

    def publish_boundary_markers(self):
        
        marker_array = MarkerArray()

        # Generate boundary points for upper and lower boundaries
        upper_points = []
        lower_points = []

        for x in self.x_range:
            y_upper = self.boundary_y_upper(x)
            y_lower = self.boundary_y_lower(x)
            
            # Clip the points to be within the specified boundaries
            y_upper = min(max(y_upper, -10.0), 10.0)
            y_lower = min(max(y_lower, -10.0), 10.0)

            upper_points.append(Point(x=x, y=y_upper, z=0.0))
            lower_points.append(Point(x=x, y=y_lower, z=0.0))

        # Create markers for the boundary lines
        upper_marker = Marker()
        upper_marker.header.frame_id = "map"
        upper_marker.header.stamp = self.get_clock().now().to_msg()
        upper_marker.ns = "boundaries"
        upper_marker.id = self.num_agents + 2  # Unique ID for boundary marker
        upper_marker.type = Marker.LINE_STRIP
        upper_marker.action = Marker.ADD
        upper_marker.scale.x = 0.05
        upper_marker.color.a = 1.0
        upper_marker.color.r = 1.0
        upper_marker.color.g = 0.0
        upper_marker.color.b = 0.0  # Red color for boundaries
        upper_marker.points = upper_points

        lower_marker = Marker()
        lower_marker.header.frame_id = "map"
        lower_marker.header.stamp = self.get_clock().now().to_msg()
        lower_marker.ns = "boundaries"
        lower_marker.id = self.num_agents + 3  # Unique ID for boundary marker
        lower_marker.type = Marker.LINE_STRIP
        lower_marker.action = Marker.ADD
        lower_marker.scale.x = 0.05
        lower_marker.color.a = 1.0
        lower_marker.color.r = 1.0
        lower_marker.color.g = 0.0
        lower_marker.color.b = 0.0  # Red color for boundaries
        lower_marker.points = lower_points

        # Create marker for the corridor
        corridor_marker = Marker()
        corridor_marker.header.frame_id = "map"
        corridor_marker.header.stamp = self.get_clock().now().to_msg()
        corridor_marker.ns = "boundaries"
        corridor_marker.id = self.num_agents + 4  # Unique ID for corridor marker
        corridor_marker.type = Marker.TRIANGLE_LIST
        corridor_marker.action = Marker.ADD
        corridor_marker.scale.x = 1.0
        corridor_marker.scale.y = 1.0
        corridor_marker.scale.z = 1.0
        corridor_marker.color.a = 0.2  # Transparency for the corridor
        corridor_marker.color.r = 0.0
        corridor_marker.color.g = 1.0
        corridor_marker.color.b = 0.0  # Green color for corridor

        for i in range(len(self.x_range) - 1):
            # Create triangles for corridor
            corridor_marker.points.append(upper_points[i])
            corridor_marker.points.append(lower_points[i])
            corridor_marker.points.append(lower_points[i + 1])
            corridor_marker.points.append(upper_points[i])
            corridor_marker.points.append(lower_points[i + 1])
            corridor_marker.points.append(upper_points[i + 1])

        # Create marker for the red zones
        red_zone_marker = Marker()
        red_zone_marker.header.frame_id = "map"
        red_zone_marker.header.stamp = self.get_clock().now().to_msg()
        red_zone_marker.ns = "red_zones"
        red_zone_marker.id = self.num_agents + 5  # Unique ID for red zone marker
        red_zone_marker.type = Marker.TRIANGLE_LIST
        red_zone_marker.action = Marker.ADD
        red_zone_marker.scale.x = 1.0
        red_zone_marker.scale.y = 1.0
        red_zone_marker.scale.z = 1.0
        red_zone_marker.color.a = 0.5  # Transparency for the red zones
        red_zone_marker.color.r = 1.0
        red_zone_marker.color.g = 0.0
        red_zone_marker.color.b = 0.0  # Red color for the zones

        # Fill the red zones (above upper and below lower boundaries)
        for i in range(len(self.x_range) - 1):
            # Upper red zone
            red_zone_marker.points.append(Point(x=float(self.x_range[i]), y=10.0, z=0.0))
            red_zone_marker.points.append(upper_points[i])
            red_zone_marker.points.append(upper_points[i + 1])
            red_zone_marker.points.append(Point(x=float(self.x_range[i + 1]), y=10.0, z=0.0))
            red_zone_marker.points.append(upper_points[i + 1])
            red_zone_marker.points.append(Point(x=float(self.x_range[i]), y=10.0, z=0.0))
            # Lower red zone
            red_zone_marker.points.append(Point(x=float(self.x_range[i]), y=-10.0, z=0.0))
            red_zone_marker.points.append(lower_points[i])
            red_zone_marker.points.append(lower_points[i + 1])
            red_zone_marker.points.append(Point(x=float(self.x_range[i + 1]), y=-10.0, z=0.0))
            red_zone_marker.points.append(lower_points[i + 1])
            red_zone_marker.points.append(Point(x=float(self.x_range[i]), y=-10.0, z=0.0))

        # Append markers to the array in the correct order to avoid overlap issues
        marker_array.markers.append(red_zone_marker)
        marker_array.markers.append(upper_marker)
        marker_array.markers.append(lower_marker)
        marker_array.markers.append(corridor_marker)

        self.publisher_boundaries.publish(marker_array)

    def boundary_y_upper(self, x):

        y = (self.boundary_a * x + self.boundary_b)**8 + self.boundary_c
        
        return min(max(y, -10.0), 10.0)  # Clipping to range [-10, 10]

    def boundary_y_lower(self, x):
        
        y = -((self.boundary_a * x + self.boundary_b)**8 + self.boundary_c)
        
        return min(max(y, -10.0), 10.0)  # Clipping to range [-10, 10]


def main(args=None):
    rclpy.init(args=args)
    robot_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    node = VisualizationNode(robot_id)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()