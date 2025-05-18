#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from localisation.msg import ArucoRange
from aruco_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import math
import tf2_ros
import tf2_geometry_msgs

class ArucoRangeNode:
    def __init__(self):
        # Initialise node
        rospy.init_node("aruco_range_node")

        # Subscribe to arucos id and pose
        rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, self.marker_callback)

        # Publisher for aruco range
        self.range_pub = rospy.Publisher("/aruco_range", ArucoRange, queue_size=10)

        # Static transform
        #self.tf_buffer = tf2_ros.Buffer()
        #self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Keep the node running
        rospy.spin()

    def marker_callback(self, msg):
        for marker in msg.markers:
            pos = marker.pose.pose.position
            dist = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2)

            range_msg = ArucoRange()
            range_msg.header = marker.header
            range_msg.aruco_id = marker.id
            range_msg.range = dist
            self.range_pub.publish(range_msg)

        """
        # Static transform
        for marker in msg.markers:
            try:
                # Transform marker pose to robot base frame
                transform = self.tf_buffer.lookup_transform(
                    target_frame="turtlebot/kobuki/base_link",  # <-- your robot base frame
                    source_frame=marker.header.frame_id,
                    #source_frame="camera_color_optical_frame"
                    time=rospy.Time(0),
                    timeout=rospy.Duration(0.5)
                )

                pose_transformed = tf2_geometry_msgs.do_transform_pose(marker.pose, transform)
                pos = pose_transformed.pose.position
                dist = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2)

                range_msg = ArucoRange()
                range_msg.header = marker.header
                range_msg.aruco_id = marker.id
                range_msg.range = dist
                self.range_pub.publish(range_msg)

            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Transform error: {e}")
        """

if __name__ == "__main__":
    try:
        ArucoRangeNode()
    except rospy.ROSInterruptException:
        pass

