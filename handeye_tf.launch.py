from launch import LaunchDescription
from launch_ros.actions import Node

# Sinh bởi handeye_solve.py | method=DANIILIDIS | RMS=3.06 mm, 0.81 deg | 60 mẫu
def generate_launch_description():
    return LaunchDescription([
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="handeye_tool0_to_camera",
             arguments=['--x', '-0.057568', '--y', '-0.094739', '--z', '0.039324', '--qx', '0.486320', '--qy', '-0.496827', '--qz', '0.497737', '--qw', '0.518569', '--frame-id', 'tool0', '--child-frame-id', 'camera_link']),
    ])
