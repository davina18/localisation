#!/usr/bin/env python3
import rospy
import sys
import termios
import tty
from geometry_msgs.msg import Twist

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def move_robot():
    pub = rospy.Publisher('/turtlebot/kobuki/commands/velocity', Twist, queue_size=10)
    rate = rospy.Rate(10)

    print("Press W A S D keys to move. Press Q to quit.")

    while not rospy.is_shutdown():
        key = get_key()
        cmd = Twist()
        if key == 'w':
            cmd.linear.x = 0.2
        elif key == 's':
            cmd.linear.x = -0.2
        elif key == 'a':
            cmd.angular.z = 0.5
        elif key == 'd':
            cmd.angular.z = -0.5
        elif key == 'q':
            print("Exiting.")
            break
        else:
            continue

        pub.publish(cmd)        # send motion
        rate.sleep()            # wait
        pub.publish(Twist())    # stop motion
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node("velocity_command")
    move_robot()
