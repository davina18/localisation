#!/usr/bin/env python3

import rospy
from localisation.msg import ArucoRange
from aruco_msgs.msg import MarkerArray
import math

class ArucoRangeNode:
    def __init__(self):
        # Initialise node
        rospy.init_node("aruco_range_node")
        
        # Subscribe to arucos id and pose
        rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, self.marker_callback)

        # Publisher for aruco range
        self.range_pub = rospy.Publisher("/aruco_range", ArucoRange, queue_size=10)

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

if __name__ == "__main__":
    try:
        ArucoRangeNode()
    except rospy.ROSInterruptException:
        pass

