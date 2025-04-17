#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
import tf.transformations as tf_trans
import math

class imuNode:
    def __init__(self):
        # Initialise node
        rospy.init_node('imu_node')

        # Subscribe to imu topic
        rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data', Imu, self.imu_callback)

        # Publisher for compass heading
        self.heading_pub = rospy.Publisher('/compass_heading', Float64, queue_size=10)

        # Keep the node running
        rospy.spin()

    def imu_callback(self, msg):
        # Extract quaternion angle from imu message
        q = msg.orientation
        quaternion = [q.x, q.y, q.z, q.w]

        # Convert angle from quaternion to yaw
        _, _, yaw = tf_trans.euler_from_quaternion(quaternion)
        yaw = self.normalise_angle(yaw)

        # Publish yaw as the compass heading
        self.heading_pub.publish(Float64(yaw))
    
    def normalise_angle(self, angle):
        # Normalise angle to (-pi, pi)
        return math.atan2(math.sin(angle), math.cos(angle))  


if __name__ == '__main__':
    try:
        imuNode()
    except rospy.ROSInterruptException:
        pass