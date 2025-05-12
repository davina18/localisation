#!/usr/bin/python3

# This simple ROS controller will expose a Twist subscriber to control the turtlebot in linear and angular velocity

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from math import sin, cos

class DiffDriveController:

    def __init__(self) -> None:

        # robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230

        self.sub = rospy.Subscriber("cmd_vel", Twist, self.cmd_vel_callback)
        self.pub = rospy.Publisher("wheel_velocities", Float64MultiArray, queue_size=10)

    def cmd_vel_callback(self, msg):
        
        v = msg.linear.x
        w = msg.angular.z

        left_wheel_velocity = (v - w * self.wheel_base_distance / 2.0) / self.wheel_radius
        right_wheel_velocity = (v + w * self.wheel_base_distance / 2.0) / self.wheel_radius

        msg = Float64MultiArray()
        msg.data = [left_wheel_velocity, right_wheel_velocity]
        self.pub.publish(msg)

if __name__ == "__main__":

    rospy.init_node("diff_drive_controller")
    controller = DiffDriveController()
    rospy.spin()
