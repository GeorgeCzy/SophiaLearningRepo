import rospy
from sophia_dev.sophia_control.llf_control import ActuatorControl

if __name__ == "__main__":
    rospy.init_node("actuator_control")
    ActuatorControl()
    rospy.spin()