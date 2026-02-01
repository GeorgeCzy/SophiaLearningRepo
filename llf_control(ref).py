import math
import threading
import time

import os
import json
import pickle

import sys
sys.path.append(os.getcwd())

import rospy
from hr_msgs.msg import TargetPosture
from geometry_msgs.msg import Vector3
from hr_msgs.srv import SetActuatorsControl, SetActuatorsControlRequest

from sophia_dev.bs_mapping.bs_mapping import convert_list_to_dict, read_map
from sophia_dev.livelinkface.receive import bind, receive_data, decode_from_livelinkface

# help function
# value boundary 
def get_bounds(limit_file_path):
    with open(limit_file_path, 'r') as limit_file:
        limits = json.load(limit_file)
    return limits

def check_bounds(name, value, limits):
    bounds = limits[name]["value"]
    if value < bounds[0]:
        value = bounds[0]
    elif value > bounds[1]:
        value = bounds[1]
    return value

# limit the change between contiguous frames
def check_contiguous(past_values, values):
    for key in values:
        if values[key]-past_values[key]>0.2:
            values[key] = past_values[key]+0.2
        elif values[key] - past_values[key]<-0.2:
            values[key] = past_values[key]-0.2
    return values

class ActuatorControl(object):
    """An example showing how to control indivisual actuators.
    """
    def __init__(self, blendshape_used = "control", configs_path = "./sophia_dev/sophia_control/configs", map_path = "./sophia_dev/bs_mapping/mapping_table/arkit_to_sophia.csv"):
        rospy.wait_for_service("/hr/actuators/get_loggers")
        
        self.pose = rospy.Publisher("/hr/actuators/pose", TargetPosture, queue_size=10)
        self.set_control = rospy.ServiceProxy("/hr/actuators/set_control", SetActuatorsControl)
        self.set_state = rospy.Publisher("/hr/animation/set_state", TargetPosture, queue_size = 10)
        self.set_neck_rotation = rospy.Publisher("/hr/animation/set_neck_rotation", Vector3, queue_size=10)
        # expression_control
        # limit the change between contiguous frames
        self.past_values = None

        # blendshape used
        with open(os.path.join(configs_path, "blendshape_used.json"), "r") as bs_used_file:
            bs_used = json.load(bs_used_file)
        self.actuator_names = bs_used[blendshape_used]
        self.ani_names = ["HeadYaw", "HeadRoll", "HeadPitch", "SneerLeft", "SneerRight"]

        # boundary
        self.limits = get_bounds(os.path.join(configs_path, "limit.json"))

        # socket
        self.sever = bind()

        # bs map
        df, columns, arkit_index, seq = read_map(map_path)
        self.df = df
        self.columns = columns
        self.arkit_index = arkit_index
        self.seq = seq

        # ros spin
        job = threading.Timer(0, self.move)
        job.daemon = True
        job.start()

    def set_manual_control(self):
        """
        Set the actuators to manual control, so them can be controlled
        directly.

        CONTROL_DISABLE: disable actuator
        CONTROL_MANUAL: control actuator by hand (code)
        CONTROL_ANIMATION: control actuator by animation
        """
        request = SetActuatorsControlRequest()
        request.control = SetActuatorsControlRequest.CONTROL_MANUAL
        request.actuators = self.actuator_names
        self.set_control(request)
   
    def move(self):
        """Move the actuators"""
        self.set_manual_control()
        
        neck_rotation = Vector3()
        neck_rotation.x = 0
        neck_rotation.y = 0
        neck_rotation.z = 0
        self.set_neck_rotation.publish(neck_rotation)
      
        pose = TargetPosture()
        ani_pose = TargetPosture()
        
        while True:
            data = receive_data(self.sever)
            data = decode_from_livelinkface(data)
            # print(data)
            if data:
                self.get_message(values=data)
                pose.names = self.actuator_names[:]
                pose.values = self.values[:]
                    
                self.pose.publish(pose)
            
                ani_pose.names = self.ani_names
                ani_pose.values = self.ani_values
            
                self.set_state.publish(ani_pose)
                        
        # conn.close()

    def get_message(self, values):
        # values: arkit params
        values = [100.0*i for i in values]
        values = convert_list_to_dict(values, self.df, self.columns, self.arkit_index, self.seq)
        print(values["UpperLidLeft"])
        # set past_values if past_values is None
        if self.past_values is None:
            self.past_values = {}
            for key in values:
                self.past_values[key] = 0
        
        # limit the change between contiguous frames
        values = check_contiguous(self.past_values, values)

        # control by animation
        self.ani_values = []
        for ani_name in self.ani_names:
            self.ani_values.append(check_bounds(ani_name, values[ani_name], self.limits))
        
        # control by actuator
        self.values = []
        for actuator_name in self.actuator_names:
            self.values.append(check_bounds(actuator_name, values[actuator_name], self.limits))
        
        self.past_values = values
        

if __name__ == "__main__":
    rospy.init_node("actuator_control")
    ActuatorControl()
    rospy.spin()