#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, ColorRGBA
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovariance
from localisation.msg import ArucoRange
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import sin, cos, atan2, sqrt
from efk import EKFNode
from GetEllipse import GetEllipse
import tf.transformations
import math

class FEKFSLAM(EKFNode):
    def __init__(self):
        super().__init__()

        # State dimensions
        self.xB_dim = 3  # Robot state: [x, y, theta]
        self.xF_dim = 2  # Feature state: [x, y]

        # Initial robot pose and covariance
        self.xk = np.zeros((self.xB_dim, 1))
        self.Pk = np.diag([0.1, 0.1, 0.1])
        self.Qk = np.diag([0.1, 0.1])  # Process noise (if not already in EKFNode)

        # Wheel parameters
        self.wheel_radius = 0.035  # meters
        self.wheel_base_distance = 0.23  # meters
        #self.Qk = np.diag([0.05, 0.05, 0.01])  # process noise covariance

        # Wheel joint names
        self.left_wheel_name = 'turtlebot/kobuki/wheel_left_joint'
        self.right_wheel_name = 'turtlebot/kobuki/wheel_right_joint'

        self.left_wheel_velocity_received = False
        self.right_wheel_velocity_received = False

        # Velocity placeholders
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_linear_velocity = 0.0
        self.right_linear_velocity = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Timing
        self.current_time = rospy.Time.now()
        self.last_time = self.current_time
        self.time = 0.0

        # Flag to wait until both velocities are received
        self.left_wheel_velocity_received = False

        # Aruco observations
        self.aruco_observations = {}  # {aruco_id: [(robot_pose, range)]}
        self.triangulation_dist_threshold = 0.3

        # Marker tracking
        self.observed_arucos = {}  # {aruco_id: (state_index, last_seen_time)}
        self.next_feature_idx = 0

        # Path tracking
        self.estimated_path = Path()
        self.estimated_path.header.frame_id = "world_ned"
        self.ground_truth_path = Path()
        self.ground_truth_path.header.frame_id = "world_ned"

        # Publishers
        self.estimated_path_pub = rospy.Publisher("/estimated_path", Path, queue_size=10)
        self.ground_truth_path_pub = rospy.Publisher("/ground_truth_path", Path, queue_size=10)
        self.estimated_path_marker_pub = rospy.Publisher("/estimated_path_marker", Marker, queue_size=10)
        self.ground_truth_path_marker_pub = rospy.Publisher("/ground_truth_path_marker", Marker, queue_size=10)
        self.ground_truth_pub = rospy.Publisher("/ground_truth_path", Path, queue_size=10)
        self.innovation_pub = rospy.Publisher("/innovation_marker", Marker, queue_size=10)
        self.landmark_pub = rospy.Publisher("/landmark_markers", MarkerArray, queue_size=10)
        self.uncertainty_pub = rospy.Publisher("/uncertainty_ellipse", Marker, queue_size=10)
        self.range_marker_pub = rospy.Publisher("/range_line_marker", Marker, queue_size=10)

        # Subscribers
        rospy.Subscriber("/aruco_range", ArucoRange, self.range_callback)
        rospy.Subscriber("/compass_heading", Float64, self.imu_yaw_callback)
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_states_callback)
        # rospy.Subscriber("/imu/data", Imu, self.imu_callback)

        # Ground truth pose tracking (optional)
        self.ground_truth_position = np.zeros(3)
        rospy.Subscriber("/turtlebot/kobuki/odom_ground_truth", Odometry, self.gt_odom_callback)

    def wrap_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def joint_states_callback(self, msg):

        for i, name in enumerate(msg.name):
            if name == self.left_wheel_name:
                self.left_wheel_velocity = msg.velocity[i]
                self.left_wheel_velocity_received = True
            elif name == self.right_wheel_name:
                self.right_wheel_velocity = msg.velocity[i]
                self.right_wheel_velocity_received = True

        if self.left_wheel_velocity_received and self.right_wheel_velocity_received:
            self.left_linear_velocity = self.left_wheel_velocity * self.wheel_radius
            self.right_linear_velocity = self.right_wheel_velocity * self.wheel_radius

            self.linear_velocity = (self.left_linear_velocity + self.right_linear_velocity) / 2
            self.angular_velocity = (self.left_linear_velocity - self.right_linear_velocity) / self.wheel_base_distance

            self.current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            if not hasattr(self, "last_time"):
                self.last_time = self.current_time
                return

            self.time = (self.current_time - self.last_time).to_sec()
            self.last_time = self.current_time

            self.xk[0, 0] += np.cos(self.xk[2, 0]) * self.linear_velocity * self.time
            self.xk[1, 0] += np.sin(self.xk[2, 0]) * self.linear_velocity * self.time
            self.xk[2, 0] = self.wrap_angle(self.xk[2, 0] + self.angular_velocity * self.time)

            uk = [self.linear_velocity, self.angular_velocity, self.time]
            self.xk, self.Pk = self.prediction(uk, self.Qk, self.xk, self.Pk)
            #rospy.loginfo(f"Updated pose: x={self.xk[0,0]:.2f}, y={self.xk[1,0]:.2f}, yaw={self.xk[2,0]:.2f}")
            if np.isnan(self.xk).any():
                rospy.logwarn("NaN detected in xk after prediction!")
            self.update_paths()

            self.left_wheel_velocity_received = False
            self.right_wheel_velocity_received = False



    '''def imu_callback(self, msg):
            """
            Callback function to handle IMU sensor data. It processes the orientation data provided in quaternion format,
            converts it to Euler angles, and updates the robot state using the yaw (orientation around the vertical axis).

            :param msg: The message received from the IMU topic, containing orientation data in quaternion format.
            """
        
            # Extract the quaternion tuple from the IMU message, which includes x, y, z, and w components.
            quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

            # Convert the quaternion to Euler angles. Since only yaw is needed for 2D motion estimation, 
            # the roll and pitch values are discarded (denoted by underscores).
            _, _, yaw_measurement = euler_from_quaternion(quaternion)
            self.yaw_measurement=yaw_measurement-np.pi/2  '''
    
    def imu_yaw_callback(self, msg):
        """
        Handle yaw data published as Float64 from /compass_heading.
        """
        yaw_meas = msg.data  # Extract the float value
        R_yaw = np.array([[0.2]])  # Adjust noise if needed
        self.xk, self.Pk = self.update_yaw(self.xk, self.Pk, yaw_meas, R_yaw)


    def gt_odom_callback(self, msg):
        """Record ground truth position for visualization"""
        self.ground_truth_position[0] = msg.pose.pose.position.x
        self.ground_truth_position[1] = msg.pose.pose.position.y
        _, _, yaw = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        self.ground_truth_position[2] = yaw
        
        # Update ground truth path
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.ground_truth_path.poses.append(pose)
        self.ground_truth_path_pub.publish(self.ground_truth_path)
        #self.publish_ground_truth_path_marker()

    def update_paths(self):
        """Update both estimated and ground truth paths"""
        #rospy.loginfo("Appending estimated pose to path")

        # Estimated path
        est_pose = PoseStamped()
        self.estimated_path.header.stamp = rospy.Time.now()
        self.estimated_path.header.frame_id = "world_ned"
        est_pose.header = self.estimated_path.header
        est_pose.pose.position.x = self.xk[0,0]
        est_pose.pose.position.y = self.xk[1,0]
        est_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, self.xk[2,0]))
        self.estimated_path.poses.append(est_pose)
        self.estimated_path_pub.publish(self.estimated_path)
        #rospy.loginfo(f"Publishing path with {len(self.estimated_path.poses)} poses")
        
        # Publish uncertainty ellipse
        self.publish_uncertainty_ellipse()
        self.publish_estimated_path_marker()
        #self.publish_ground_truth_path_marker()

    def publish_uncertainty_ellipse(self):
        """Visualize robot pose uncertainty as a 2D ellipse using GetEllipse()"""
        cov = self.Pk[0:2, 0:2]
        center = self.xk[0:2, 0].reshape(2, 1)
        ellipse_points = GetEllipse(center, cov, sigma=3)

        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot_uncertainty"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # Line thickness
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0) 
        marker.pose.orientation.w = 1.0
        marker.pose.position.z = 0.0

        for i in range(ellipse_points.shape[1]):
            pt = Point()
            pt.x = ellipse_points[0, i]
            pt.y = ellipse_points[1, i]
            pt.z = 0.0
            marker.points.append(pt)

        # Close the loop
        pt0 = Point()
        pt0.x = ellipse_points[0, 0]
        pt0.y = ellipse_points[1, 0]
        pt0.z = 0.0
        marker.points.append(pt0)

        self.uncertainty_pub.publish(marker)

    def publish_estimated_path_marker(self):

        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "estimated_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width in meters
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0

        for pose in self.estimated_path.poses:
            pt = Point()
            pt.x = pose.pose.position.x
            pt.y = pose.pose.position.y
            pt.z = 0.0
            marker.points.append(pt)

        self.estimated_path_marker_pub.publish(marker)

    def publish_ground_truth_path_marker(self):
        if len(self.ground_truth_path.poses) < 2:
            return

        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "ground_truth_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width in meters
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0

        for pose in self.ground_truth_path.poses:
            pt = Point()
            pt.x = pose.pose.position.x
            pt.y = pose.pose.position.y
            pt.z = 0.0
            marker.points.append(pt)

        self.ground_truth_path_marker_pub.publish(marker)


    def g(self, xBk, range):
        """Initialize new feature position"""
        # im applying here : xf =xr+r⋅cos(θ)  ,  yf = yr+r⋅sin(θ) this is the initial guess for the landmark's position
        # if the robot initally is facing angle theta and we place the ladnamrk at xf and yf then we are guessing that the landmark is in front of the robot
        # which ok but the real one could be anywhere -> bearing ambiguity
        # If we assume small uncertainty (like tight covariance) in the feature's 2D position, your EKF believes it's very confident but it's not

        theta = xBk[2, 0]
        x = xBk[0, 0] + range * cos(theta)
        y = xBk[1, 0] + range * sin(theta)
        return np.array([[x], [y]])

    def Jgx(self, xBk, range):
        """Jacobian of g w.r.t. robot state"""
        theta = xBk[2, 0]
        return np.array([
            [1, 0, -range * sin(theta)],
            [0, 1, range * cos(theta)]
        ])

    def Jgv(self, xBk, range):
        """Jacobian of g w.r.t. measurement"""
        theta = xBk[2, 0]
        return np.array([
            [cos(theta)],
            [sin(theta)]
        ])

    def h_range(self, x_r, y_r, x_f, y_f):
        """Range measurement model"""
        dx = x_f - x_r
        dy = y_f - y_r
        return sqrt(dx**2 + dy**2)

    def range_callback(self, msg):
        """Handle new range measurement"""
        aruco_id = msg.aruco_id
        range_measurement = msg.range
        current_pose = self.xk[0:3, 0].copy()  # [x, y, theta]

        # If the aruco hasn't been observed before, initialise an observation list for it
        if aruco_id not in self.aruco_observations:
            self.aruco_observations[aruco_id] = []

        # Add the robots current pose and range measurement to the arucos observation list
        self.aruco_observations[aruco_id].append((current_pose, range_measurement))

        observations = self.aruco_observations[aruco_id]
        # If at least 2 observations of this aruco have been made
        if len(observations) >= 2:
            # Compute the Euclidean distance between the first pose and most recent pose 
            p1, _ = observations[0]
            p2, _ = observations[-1]
            dist = np.linalg.norm(p2[0:2] - p1[0:2])
            # Only triangulate if lower than the distance threshold
            if dist >= self.triangulation_dist_threshold:
                if aruco_id in self.observed_arucos:
                    return # aruco already added to state
                # Estimate the position of the landmark using triangulation
                landmark_pos = self.triangulate(observations)
                # Report triangulation failure
                if landmark_pos is None:
                    rospy.logwarn(f"Triangulation failed for Aruco ID {aruco_id}")
                    return None
                # Manually set an initial covariance for the new landmark
                cov = np.diag([0.2, 0.2])
                # Print uncertainty ellipse area before new landmark addition
                area_before = np.pi * np.sqrt(np.linalg.det(self.Pk[0:2, 0:2]))
                # Add the new landmark to the robot state using the estimated pos and cov
                self.add_new_landmark(aruco_id, landmark_pos, cov)
                # Update step
                Rn = np.array([[0.1]])  # measurement noise
                self.xk, self.Pk = self.update(self.xk, self.Pk, range_measurement, Rn, aruco_id)
                # Print uncertainty ellipse area after new landmark addition
                area_after = np.pi * np.sqrt(np.linalg.det(self.Pk[0:2, 0:2]))
                rospy.loginfo(f"Uncertainty ellipse area before: {area_before:.4f}, after: {area_after:.4f}")
                self.visualize_range_circle(np.linalg.norm(landmark_pos - self.xk[0:2, 0].reshape(2,1)))
                del self.aruco_observations[aruco_id]
                rospy.loginfo(f"Added landmark {aruco_id} at {landmark_pos.ravel()} with covariance {np.diag(cov)}")

    def triangulate(self, observations):
        # Estimate landmark position using least-squares
        A = []
        b = []
        # For each observation of this aruco
        for (pose, r) in observations:
            x, y, theta = pose
            xi = x + r * np.cos(theta) # estimated x-position of the landmark
            yi = y + r * np.sin(theta) # estimated y-position of the landmark

            # Build linear system using the circle equation
            A.append([-2 * xi, -2 * yi])
            b.append(-(xi**2 + yi**2 - r**2))

        A = np.array(A) 
        b = np.array(b).reshape(-1, 1)

        # Solve the least squares problem: A * [x; y] = b
        landmark_est = np.linalg.lstsq(A, b, rcond=None)[0]

        return landmark_est
    
    def add_new_landmark(self, aruco_id, position, covariance):
        # Skip if the landmark has already been added to the state
        if aruco_id in self.observed_arucos:
            return

        # Add the new landmarks position to the state vector
        self.xk = np.vstack([self.xk, position.reshape(2, 1)])
        # Initialise cross-covariance to 0 (uncorrelated initially)
        cross_cov = np.zeros((self.Pk.shape[0], 2))
        # Expand covariance matrix to include the new landmark
        self.Pk = np.block([
            [self.Pk, cross_cov],
            [cross_cov.T, covariance]
        ])

        # Log observation of aruco
        self.observed_arucos[aruco_id] = (self.next_feature_idx, rospy.Time.now().to_sec())
        self.next_feature_idx += 1


    # ---------------------- SLAM Functions ----------------------

    def AddNewFeatures(self, xk, Pk, znp, Rnp, aruco_id):
        """Initialize new features in the state vector"""
        xk_plus = xk.copy()
        Pk_plus = Pk.copy()
        PBk = Pk[0:self.xB_dim, 0:self.xB_dim]

        # Convert scalar inputs to arrays if needed
        if isinstance(znp, (int, float)):
            znp = np.array([znp])
            Rnp = np.array([Rnp])
            aruco_id = [aruco_id]

        for i in range(len(znp)):
            current_id = aruco_id[i]
            if current_id in self.observed_arucos:
                continue  # Skip if already known

            # Initialize new feature
            range = znp[i]
            xfi = self.g(xk[0:self.xB_dim], range)

            # Add to state vector
            xk_plus = np.vstack([xk_plus, xfi.reshape(-1, 1)])  # xfi must be (2,1)
            self.observed_arucos[current_id] = self.next_feature_idx
            self.next_feature_idx += 1

            # Update covariance
            Jgxi = self.Jgx(xk[0:self.xB_dim], range)
            Jgvi = self.Jgv(xk[0:self.xB_dim], range)

            # New feature covariance
            Pfi = Jgxi @ PBk @ Jgxi.T + Jgvi @ Rnp[i] @ Jgvi.T

            # Cross-covariance terms
            Pcross = np.hstack([
                Jgxi @ Pk[0:self.xB_dim, 0:self.xB_dim],
                Jgxi @ Pk[0:self.xB_dim, self.xB_dim:]]
            )

            # Expand covariance matrix
            Pk_plus = np.block([
                [Pk_plus, Pcross.T],
                [Pcross, Pfi]
            ])

        return xk_plus, Pk_plus

    def update(self, xk, Pk, zn, Rn, aruco_id):
        """Main EKF update with range measurements"""
        # Convert scalar to array if needed
        if isinstance(zn, (int, float)):
            zn = np.array([[zn]])

        # Check if this is a new feature
        #if aruco_id not in self.observed_arucos:
        #    return self.AddNewFeatures(xk, Pk, zn, Rn, aruco_id)

        # Existing feature update
        #idx = (len(self.observed_arucos))
        idx, _ = self.observed_arucos[aruco_id]
        start_idx = self.xB_dim + idx * self.xF_dim
        self.observed_arucos[aruco_id] = idx

        # Get current states
        x_r, y_r, theta_r = xk[0:3, 0]
        x_f, y_f = xk[start_idx:start_idx+2, 0]

        # Expected measurement and Jacobian
        h = self.h_range(x_r, y_r, x_f, y_f)
        H = np.zeros((1, len(xk)))
        
        # Range derivatives
        dx = x_f - x_r
        dy = y_f - y_r
        dist = max(np.sqrt(dx**2 + dy**2), 1e-6)  # Avoid division by zero

        # Jacobian components
        H[0, 0] = -dx/dist  
        H[0, 1] = -dy/dist  
        H[0, 2] = 0         
        H[0, start_idx] = dx/dist    
        H[0, start_idx+1] = dy/dist 

        # Kalman update
        innovation = zn - h
        S = H @ Pk @ H.T + Rn
        K = Pk @ H.T @ np.linalg.inv(S)
        
        xk = xk + K @ innovation
        Pk = (np.eye(len(xk)) - K @ H) @ Pk

        # Visualize innovation
        self.visualize_innovation(innovation, (x_r, y_r, 0))

        return xk, Pk

    def update_yaw(self, xk, Pk, yaw_meas, R_yaw):
        """Update state with IMU yaw measurement"""
        # Normalize angles to [-pi, pi]
        yaw_pred = xk[2, 0]
        innovation = np.array([[yaw_meas - yaw_pred]])
        innovation[0, 0] = atan2(sin(innovation[0, 0]), cos(innovation[0, 0]))

        # Jacobian (only affects yaw)
        H = np.zeros((1, len(xk)))
        H[0, 2] = 1  # Only the yaw component

        # Kalman update
        S = H @ Pk @ H.T + R_yaw
        K = Pk @ H.T @ np.linalg.inv(S)
        
        xk = xk + K @ innovation
        Pk = (np.eye(len(xk)) - K @ H) @ Pk

        return xk, Pk
    
    # ---------------------- Visualization ----------------------

    def visualize_range_circle(self, range_val):
        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(0.2)  # Auto-remove after 0.2s

        marker.ns = "range_circles"
        marker.id = 1000
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color = ColorRGBA(0.2, 0.8, 1.0, 0.8)

        # Generate circle points
        angles = np.linspace(0, 2 * np.pi, 50)
        for a in angles:
            pt = Point()
            pt.x = self.xk[0, 0] + range_val * np.cos(a)
            pt.y = self.xk[1, 0] + range_val * np.sin(a)
            pt.z = 0.0
            marker.points.append(pt)

        marker.points.append(marker.points[0])  # Close the loop
        self.range_marker_pub.publish(marker)


    def visualize_innovation(self, innovation, position, marker_id=0):
        """Visualize innovation as text marker"""
        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "innovations"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2] + 0.4
        marker.scale.z = 0.2  # Text height
        error = abs(innovation[0, 0])
        if error < 0.2:
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green (low error)
        elif error < 0.5:
            marker.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)  # Yellow
        else:
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red (high error)

        marker.text = f"Innov: {innovation[0,0]:.3f}"
        self.innovation_pub.publish(marker)

    def visualize_all_landmarks(self):
        """Visualize all landmarks with uncertainty ellipses"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for aruco_id, (idx, _) in self.observed_arucos.items():
            start_idx = self.xB_dim + idx * self.xF_dim
            x, y = self.xk[start_idx:start_idx+2, 0]
            cov = self.Pk[start_idx:start_idx+2, start_idx:start_idx+2]
            
            # Landmark marker
            marker = Marker()
            marker.header.frame_id = "world_ned"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "landmarks"
            marker.id = aruco_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
            
            # Uncertainty ellipse for landmark
            ellipse_points = GetEllipse(np.array([[x], [y]]), cov, sigma=3)

            ellipse = Marker()
            ellipse.header = marker.header
            ellipse.ns = "landmark_uncertainty"
            ellipse.id = aruco_id + 1000
            ellipse.type = Marker.LINE_STRIP
            ellipse.action = Marker.ADD
            ellipse.scale.x = 0.02  # Line thickness
            ellipse.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Blue color
            ellipse.pose.orientation.w = 1.0  # Identity rotation
            ellipse.pose.position.z = 0.0

            # Add all ellipse points
            for i in range(ellipse_points.shape[1]):
                pt = Point()
                pt.x = ellipse_points[0, i]
                pt.y = ellipse_points[1, i]
                pt.z = 0.0
                ellipse.points.append(pt)

            # Optionally close the ellipse
            pt0 = Point()
            pt0.x = ellipse_points[0, 0]
            pt0.y = ellipse_points[1, 0]
            pt0.z = 0.0
            ellipse.points.append(pt0)

                        
            # Landmark ID text
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "landmark_ids"
            text_marker.id = aruco_id + 2000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = 0.3
            text_marker.scale.z = 0.2
            text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            text_marker.text = str(aruco_id)
            
            marker_array.markers.append(marker)
            marker_array.markers.append(ellipse)
            marker_array.markers.append(text_marker)

        self.landmark_pub.publish(marker_array)

   

if __name__ == '__main__':
    rospy.init_node('fekf_slam_node')
    try:
        slam = FEKFSLAM()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("FEKF-SLAM node terminated")