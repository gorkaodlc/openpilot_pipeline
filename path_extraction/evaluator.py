# ---------------- TEMPLATE ---------------------------------------
# This is a template to help you start writing PythonBridge code  -
# -----------------------------------------------------------------

import rtmaps.core as rt
import rtmaps.types
from rtmaps.base_component import BaseComponent  # base class
import numpy as np
import utm
import math
from matplotlib import pyplot as plt
import cv2

def get_local_trajectory(img_ts, gt_waypoints):
    # Get closest GPS point
    row_index = np.argmin(np.abs(gt_waypoints[:, 0] - img_ts))
    # Only consider future waypoints
    gt_waypoints = gt_waypoints[row_index:row_index+100*20, :]
    # Get current position
    ts, lat, lon, alt, heading = gt_waypoints[0, :]
    x0, y0, zone_number, zone_letter = utm.from_latlon(lat, lon)
    # Transform GPS points to local coordinates in current frame
    x0_all, y0_all, _, _ =utm.from_latlon(gt_waypoints[:,1], gt_waypoints[:,2], force_zone_number=zone_number, force_zone_letter=zone_letter)
    x0_all -= x0
    y0_all -= y0
    # Adjust depending on heading
    yaw = math.radians(90 - heading)
    yaw_to_adjust = -yaw
    R = np.array([[math.cos(yaw_to_adjust), -math.sin(yaw_to_adjust)], 
                  [math.sin(yaw_to_adjust), math.cos(yaw_to_adjust)]])
    xy_rotated = R @ np.vstack((x0_all, y0_all))
    return xy_rotated

# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    
    # Constructor has to call the BaseComponent parent class
    def __init__(self):
        BaseComponent.__init__(self)  # call base class constructor

    # Dynamic is called frequently:
    # - When loading the diagram
    # - When connecting or disconnecting a wire
    # Here you create your inputs, outputs and properties
    def Dynamic(self):
        # Adding an input called "in" of ANY type
        self.add_input("in", rtmaps.types.ANY)  # define an input

        self.add_property("ground_truth_trajectory", "", subtype=rtmaps.types.FILE)

        # Define the output. The type is set to AUTO which means that the output will be typed automatically.
        # You don’t need to set the buffer_size, in that case it will be set automatically.
        #self.add_output("out", rtmaps.types.AUTO)

        
        
# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Passing through Birth()")
        # Load Ground truth trajectory
        gt_path = self.get_property("ground_truth_trajectory")
        self.gt_waypoints = np.loadtxt(gt_path, delimiter=';')
        self.frame_size = (800, 600)
        self.video_writer = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, self.frame_size)

# Core() is called every time you have a new inputs available, depending on your chosen reading policy
    def Core(self):
        img           = self.inputs["in"].ioelt.data.image_data # Image in RGB
        img_ts        = self.inputs["in"].ioelt.ts
        gt_trajectory = get_local_trajectory(img_ts, self.gt_waypoints)

        # TODO: Get trajectory from supercombo
        # TODO: Compare supercombo trajectory with ground truth trajectory


        # Just copy the input to the output here
        #out = self.inputs["in"].ioelt
        #self.write("out", out)

# Death() will be called once at diagram execution shutdown
    def Death(self):
        self.video_writer.release()
        print("Passing through Death()")
