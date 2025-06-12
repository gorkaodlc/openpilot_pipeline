# Supercombo evaluation
Pipeline to evaluate the Supercombo model (or any other lane center/trajectory estimation). The estimation is compared to the actual trajectory of the vehicle (using its GPS coordinates).

# Steps
First, the GPS recording from RTMaps needs to be processed (gps_preprocess/preprocess.rtd), in order to have the whole GPS trajectory before the execution of the model in each frame.

Then, the "evaluation.rtd" diagram is executed (remember to update the GPS trajectory file and the RTMaps recording path in the diagram).