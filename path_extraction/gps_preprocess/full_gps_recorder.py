# ---------------- TEMPLATE ---------------------------------------
# This is a template to help you start writing PythonBridge code  -
# -----------------------------------------------------------------

import rtmaps.core as rt
import rtmaps.types
from rtmaps.base_component import BaseComponent  # base class
import numpy as np

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
        self.add_input("lat_lon_alt", rtmaps.types.ANY)  # define an input
        self.add_input("gps_heading", rtmaps.types.ANY)

        # Define the output. The type is set to AUTO which means that the output will be typed automatically.
        # You don’t need to set the buffer_size, in that case it will be set automatically.
        self.add_output("out", rtmaps.types.AUTO)
        
# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Passing through Birth()")

# Core() is called every time you have a new inputs available, depending on your chosen reading policy
    def Core(self):
        lat_lon_alt = self.inputs["lat_lon_alt"].ioelt.data
        heading = self.inputs["gps_heading"].ioelt.data
        out = rtmaps.types.Ioelt()
        out.data = np.append(lat_lon_alt, heading)
        out.ts = self.inputs["lat_lon_alt"].ioelt.ts
        self.write("out", out)

# Death() will be called once at diagram execution shutdown
    def Death(self):
        print("Passing through Death()")
