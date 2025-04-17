#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from localisation.msg import ArucoRange
import math

class ArucoRangeNode:
    def __init__(self):
        # Initialise node
        rospy.init_node("aruco_range_node")

        # Subscribe to aruco pose
        rospy.Subscriber("/aruco_single/pose", PoseStamped, self.pose_callback)

        # Subscribe to aruco id
        rospy.Subscriber("/aruco_single/marker", Int32, self.id_callback)

        # Publisher for aruco range
        self.range_pub = rospy.Publisher("/aruco_range", ArucoRange, queue_size=10)

        # Stores the latest aruco id
        self.latest_id = None 

        # Keep the node running
        rospy.spin()

    def id_callback(self, msg):
        self.latest_id = msg.data

    def pose_callback(self, msg):
        # Skip if no marker has been detected yet
        if self.latest_id is None:
            return

        # Calculate Euclidean distance between camera and marker
        pose = msg.pose.position
        distance = math.sqrt(pose.x**2 + pose.y**2 + pose.z**2)

        # Log aruco detection
        rospy.loginfo(f"Aruco ID {self.latest_id} detected at distance {distance:.2f}m")

        # Publish custom ArucoRange msg that contains both id and range
        range_msg = ArucoRange()
        range_msg.header = msg.header
        range_msg.aruco_id = self.latest_id
        range_msg.range = distance

        self.range_pub.publish(range_msg)

if __name__ == "__main__":
    try:
        ArucoRangeNode()
    except rospy.ROSInterruptException:
        pass
