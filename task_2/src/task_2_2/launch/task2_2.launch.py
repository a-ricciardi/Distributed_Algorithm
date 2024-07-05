import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='task_2_2',
            executable='optimization_node',
            name='optimization_node'
        ),
        Node(
            package='task_2_2',
            executable='visualization_node',
            name='visualization_node_0',
            arguments=['0']
        ),
        Node(
            package='task_2_2',
            executable='visualization_node',
            name='visualization_node_1',
            arguments=['1']
        ),
        Node(
            package='task_2_2',
            executable='visualization_node',
            name='visualization_node_2',
            arguments=['2']
        ),
        Node(
            package='task_2_2',
            executable='visualization_node',
            name='visualization_node_3',
            arguments=['3']
        ),
        Node(
            package='task_2_2',
            executable='visualization_node',
            name='visualization_node_4',
            arguments=['4']
        ),
        Node(
            package='task_2_2',
            executable='visualization_node',
            name='visualization_node_5',
            arguments=['5']
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(get_package_share_directory('task_2_2'), 'config', 'default.rviz')],
            output='screen'
        )
    ])
