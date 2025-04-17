#!/usr/bin/env python3

# Publish velocity command:
# rostopic pub /turtlebot/kobuki/commands/velocity geometry_msgs/Twist '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'

import rospy
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
import tf2_ros
import math
import numpy as np

class WheelOdomNode:
    def __init__(self):
        # Initialise node
        rospy.init_node('wheel_odom_node')

        # Robot parameters
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.035)  # in meters
        self.wheel_base = rospy.get_param('~wheel_base', 0.23)       # in meters

        # Initialise robot pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Wheel positions and timestamps
        self.last_time = None                   # Timestamp of last update
        self.last_left_pos = None               # Last updated left wheel position
        self.last_right_pos = None              # Last updated right wheel position
        self.latest_left = None                 # Latest left wheel position
        self.latest_right = None                # Latest right wheel position
        self.left_time = None                   # Timestamp of latest left wheel position
        self.right_time = None                  # Timestamp of latest right wheel position

        # Publishers
        self.odom_pub = rospy.Publisher('/wheel_odom', Odometry, queue_size=10)  # Odometry publisher
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()                     # tf publisher

        # Subscribe to joint states
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_state_callback) 

        # Keep the node running
        rospy.spin()                             

    def joint_state_callback(self, msg):
        # Loop over joints until wheel joints
        for i, name in enumerate(msg.name):
            if name == "turtlebot/kobuki/wheel_left_joint":
                self.latest_left = msg.position[i]
                self.left_time = msg.header.stamp
            elif name == "turtlebot/kobuki/wheel_right_joint":
                self.latest_right = msg.position[i]    
                self.right_time = msg.header.stamp    

        # Handle no data
        if self.latest_left is None or self.latest_right is None:
            return
        if self.left_time is None or self.right_time is None:
            return

        # Use the newest timestamp
        current_time = max(self.left_time, self.right_time)

        # Handle first callback
        if self.last_time is None:
            self.last_time = current_time
            self.last_left_pos = self.latest_left
            self.last_right_pos = self.latest_right
            return

        # Time difference
        dt = (current_time - self.last_time).to_sec()
        if dt == 0:
            return  # handle zero division

        # Calculate wheel displacements
        d_left = self.wheel_radius * (self.latest_left - self.last_left_pos)
        d_right = self.wheel_radius * (self.latest_right - self.last_right_pos)

        # Change in displacement and orientation
        d_forward = (d_left + d_right) / 2.0            # distance moved forwards
        d_theta = (d_right - d_left) / self.wheel_base  # change in orientation

        # Update orientation
        self.theta += d_theta
        self.theta = self.normalise_angle(self.theta)

        # Update position
        self.x += d_forward * math.cos(self.theta)
        self.y += d_forward * math.sin(self.theta)

        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = "world_ned"
        odom_msg.child_frame_id = "turtlebot/kobuki/base_footprint"
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = self.yaw_to_quaternion(self.theta)
        odom_msg.twist.twist.linear.x = d_forward / dt
        odom_msg.twist.twist.angular.z = d_theta / dt
        odom_msg.pose.covariance = [0.01] * 36    # placeholder for uncertainty in position and orientation (6x6)
        odom_msg.twist.covariance = [0.01] * 36   # placeholder for uncertainty in linear and angular velocity (6x6)

        # Publish odometry
        self.odom_pub.publish(odom_msg)

        # Create tf message
        tf_msg = TransformStamped()
        tf_msg.header.stamp = current_time
        tf_msg.header.frame_id = "world_ned"
        tf_msg.child_frame_id = "turtlebot/kobuki/base_footprint"
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = odom_msg.pose.pose.orientation

        # Send tf
        self.tf_broadcaster.sendTransform(tf_msg)

        # Update last values
        self.last_time = current_time
        self.last_left_pos = self.latest_left
        self.last_right_pos = self.latest_right

    def yaw_to_quaternion(self, yaw):
        # Convert angle from yaw to quaternion
        return Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )

    def normalise_angle(self, angle):
        # Normalise angle to (-pi, pi)
        return math.atan2(math.sin(angle), math.cos(angle))      


if __name__ == '__main__':
    try:
        WheelOdomNode()
    except rospy.ROSInterruptException:
        pass