#!/usr/bin/env python3
import math
import rospy
import tf
import numpy as np
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState, Imu
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point

class EKFNode:
    def __init__(self):
        #rospy.init_node('EKF_node')
        
        # Initialize parameters with proper values
        self.xB_dim = 3
        self.xF_dim = 2
        self.wheel_radius = 0.035  # in meters
        self.wheel_base_distance = 0.23  # in meters
        
        # State variables
        self.xk_1 = np.zeros((self.xB_dim, 1))
        self.Pk_1 = np.diag([0.1, 0.1, 0.1])
        self.Qk = np.diag([0.1, 0.1, 0.1])
        
        # Current time
        self.current_time = rospy.Time.now()
        self.last_time = self.current_time
        
        # Publishers
        self.odom_pub = rospy.Publisher('/wheel_odom', Odometry, queue_size=10)
        self.odom_broadcaster = tf.TransformBroadcaster()
        self.path_pub = rospy.Publisher('/path', Path, queue_size=10)
        self.path_marker_pub = rospy.Publisher('/path_marker', Marker, queue_size=10)
        self.marker_pub_odom = rospy.Publisher('/odom_uncertainity', Marker, queue_size=10)
        
        # Path initialization
        self.path = Path()
        self.path.header.frame_id = "world_ned"
        
        # Subscribers
        #rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_state_callback)


    '''def odom_callback(self, msg):
        self.current_time = msg.header.stamp
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        
        if self.last_time is None:
            self.last_time = self.current_time
            return

        dt = (self.current_time - self.last_time).to_sec()
        if dt <= 0:
            return

        uk = [v, w, dt]
        xk_bar, Pk_bar = self.prediction(uk, self.Qk, self.xk_1, self.Pk_1)

        self.xk_1 = xk_bar
        self.Pk_1 = Pk_bar
        self.last_time = self.current_time'''

    def prediction(self, uk, Qk, xk, Pk):
        v = uk[0]
        w = uk[1]
        t = uk[2]
        theta = xk[2, 0]
        
        # Initialize xk_bar properly
        xk_bar = xk.copy()
        xk_bar[0, 0] += np.cos(theta) * v * t
        xk_bar[1, 0] += np.sin(theta) * v * t
        xk_bar[2, 0] = (theta + w * t + np.pi) % (2 * np.pi) - np.pi  # wrap angle
            
        # Jacobians w.r.t robot state (F1) and control input noise (F2)
        F1 = np.array([
            [1.0, 0.0, -np.sin(theta) * v * t],
            [0.0, 1.0,  np.cos(theta) * v * t],
            [0.0, 0.0,  1.0]
        ])

        F2 = np.array([
            [np.cos(theta) * t, 0],
            [np.sin(theta) * t, 0],
            [0, t]
        ])
        
        # Extend Jacobians to match state size
        n = len(xk)  # full state length
        Pxx = np.eye(n)
        Pyy = np.zeros((n, 2))  # Qk must be 2x2: noise for [v, w]

        Pxx[:3, :3] = F1
        Pyy[:3, :2] = F2

        # Update the state covariance matrix
        Pk_bar = Pxx @ Pk @ Pxx.T + Pyy @ Qk @ Pyy.T
        
        # Publish odometry
        self.publish_odometry(xk_bar, Pk_bar)

        return xk_bar, Pk_bar

    def publish_odometry(self, xk, Pk):
        q = quaternion_from_euler(0, 0, xk[2,0])
        
        odom = Odometry()
        odom.header.stamp = self.current_time
        odom.header.frame_id = "world_ned"
        odom.child_frame_id = "turtlebot/kobuki/base_footprint"
        
        odom.pose.pose.position.x = xk[0,0]
        odom.pose.pose.position.y = xk[1,0]
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        # Covariance matrix
        odom.pose.covariance = [0.0] * 36
        # x row
        odom.pose.covariance[0]  = Pk[0, 0]  # x–x (uncertainty in x)
        odom.pose.covariance[1]  = Pk[0, 1]  # x–y (x-y covariance)
        odom.pose.covariance[5]  = Pk[0, 2]  # x–yaw (x-yaw covariance)

        # y row
        odom.pose.covariance[6]  = Pk[1, 0]  # y–x (y-x covariance)
        odom.pose.covariance[7]  = Pk[1, 1]  # y–y (uncertainty in y)
        odom.pose.covariance[11] = Pk[1, 2]  # y–yaw (y-yaw covariance)

        # yaw row
        odom.pose.covariance[30] = Pk[2, 0]  # yaw–x (yaw-x covariance)
        odom.pose.covariance[31] = Pk[2, 1]  # yaw–y (yaw-y covariance)
        odom.pose.covariance[35] = Pk[2, 2]  # yaw–yaw (uncertainty in yaw)

        
        self.odom_pub.publish(odom)
        self.odom_broadcaster.sendTransform(
            (xk[0,0], xk[1,0], 0.0),
            q,
            rospy.Time.now(),
            odom.child_frame_id,
            odom.header.frame_id
        )

'''if __name__ == '__main__':
    try:
        EKFNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass'''