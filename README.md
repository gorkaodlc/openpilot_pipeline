# OpenPilot Supercombo Pipeline

This repository contains scripts and notebooks to run the **Supercombo** model from [openpilot](https://github.com/commaai/openpilot) with our own camera setup. The ONNX version of the model is not tracked in this repository due to its size. Obtain `supercombo.onnx` from an openpilot checkout and place it under `common/models/` before running any code.

![Path prediction](train/outs/only_path.gif)

## Directory Overview
- **train/** – training utilities and the `inference.ipynb` notebook
- **inference/** – real-time inference script
- **path_extraction/** – tools for GPS processing and evaluation
- **rtmaps/** – RTMaps diagrams for recording, undistortion and inference
- **gt_distill/** – create ground‑truth predictions from a pre-trained model
- **gt_real/** – placeholder for creating ground truth using sensors
- **common/** – openpilot modules and camera configuration
- **doc/** – architecture images

## Setup
1. Install the dependencies with conda:
   ```bash
   conda env create -f environment.yml
   ```
2. Copy or export the `supercombo.onnx` model into `common/models/`.
3. Adjust camera intrinsics and other parameters in `common/config.py` to match your hardware.

## Running Inference
The easiest way to test the model is the Jupyter notebook located at [`train/inference.ipynb`](train/inference.ipynb). It shows how to load a video, run the model and visualize lane lines and the predicted path.

For real-time inference from a video file or camera feed you can run:
```bash
python inference/openpilot_realtime_inf.py
```

## Evaluating Predictions
To compare the model output with the real vehicle trajectory:
1. Preprocess the GPS log with the RTMaps diagram `path_extraction/gps_preprocess/preprocess.rtd`. This creates a record containing the ground truth path.
2. Run `rtmaps/inference_from_record.rtd` to execute the model on the recorded video and compute per‑frame metrics. The same metrics can also be obtained with the Python script `inference/openpilot_realtime_inf.py`.

The input video should have a resolution of **1160 × 874** and be recorded with the camera as parallel to the ground as possible.

## Training
The scripts in `train/` allow fine‑tuning or distilling Supercombo. Launch `train/train.py` with a dataset in comma2k19 format. After training, `torch_to_onnx.py` converts the resulting PyTorch model back to ONNX so it can be used with the inference tools.

## Configuration
All calibration parameters (focal length, camera height, etc.) are defined in [`common/config.py`](common/config.py). Edit this file to adapt the model to your specific camera placement.

## Supercombo model info
### Input


#### 1. Image Stream

Two consecutive images captured at a frequency of 20 Hz are input to the model.  
Each image has a resolution of **256 × 512 pixels** in **RGB format**.

The images are represented in **YUV420** format with **6 channels**, organized as follows:

- **Channels 0, 1, 2, and 3**: Contain high-resolution **Y (luminance)** information, arranged using the following sampling pattern from a NumPy array:

  ```
  Y[::2, ::2]  
  Y[::2, 1::2]  
  Y[1::2, ::2]  
  Y[1::2, 1::2]
  ```

- **Channel 4**: Contains medium-resolution **U (chrominance)** information.  
- **Channel 5**: Contains medium-resolution **V (chrominance)** information.

The total size of this input is **393,216 float32 values**.

#### 2. Desire Vector

A **one-hot encoded vector** that tells the model which action is desired.  
This vector only needs to be sent **in a single frame**.  
It has a size of **8 float32 values**.

### 3. Traffic Convention

A **one-hot encoded vector** indicating whether traffic follows the **right-hand** or **left-hand** side of the road.  
This vector has a size of **2 float32 values**.

#### 4. Recurrent State
A **recurrent state vector** used as input to the **GRU (Gated Recurrent Unit)** to provide temporal context to the model.  
This vector has a size of **512 float32 values**.

### Outputs
The **Supercombo model** generates multiple outputs, including predictions for vehicle trajectory, lane and object detection, and internal states of the model.

#### 1. Trajectory Planning (plan)

- Contains **5 trajectory hypotheses**, each with **991 values**, totaling **4955 values**.
- Each hypothesis includes mean and standard deviation of several variables over 33 time steps (t = 0s to t = 10s):
  - Position (x, y, z) in the current frame (meters)
  - Velocity (x, y, z) in local frame (m/s)
  - Acceleration (x, y, z) in local frame (m/s²)
  - Roll, pitch, yaw angles (radians)
  - Roll, pitch, yaw angular velocities (radians/s)
- Includes 1 probability value indicating the likelihood of each plan hypothesis.
- **Total size:** 4955 float32 values

#### 2. Lane Line Detection (lanelines)

- Prediction of **4 lane lines** (outer left, left, right, outer right).
- Each line has **132 values** (mean and std of y and z across 33 x positions).
- **Total size:** 528 float32 values

#### 3. Lane Line Probabilities

- **Two probabilities** for each of the 4 lane lines.
- **Total size:** 8 float32 values

#### 4. Road Edges

- Prediction of **2 road edges** (left and right).
- Each edge has **132 values** (mean and std of y and z across 33 x positions).
- **Total size:** 264 float32 values

#### 5. Lead Vehicle Predictions (leads)

- Prediction of **2 hypotheses** for vehicles in the path.
- Each hypothesis contains **51 values**, totaling **102 values**:
  - Position (x, y), velocity, and acceleration of the lead vehicle at 6 time steps (0, 2, 4, 6, 8, 10s)
  - Probability of each hypothesis being the most likely at 0s, 2s, and 4s
- **Total size:** 102 float32 values

#### 6. Lead Vehicle Probabilities

- Probability of a lead vehicle being present at 0s, 2s, and 4s.
- **Total size:** 3 float32 values

#### 7. Desire State

- Probability of the model executing one of the 8 possible desire actions.
- **Total size:** 8 float32 values

#### 8. Scene Metadata (meta 2)

- Includes:
  - Probability that OpenPilot is engaged (1 value)
  - Probabilities of events in the next 10 seconds (35 values), such as:
    - OpenPilot disengaging via throttle, brake, or steering
    - Sudden decelerations (-3, -4, -5 m/s²)
  - Probabilities of active blinkers in the next 10 seconds (12 values)
  - Probabilities of each of the 8 desire actions at different times (32 values)
- **Total size:** 80 float32 values

#### 9. Vehicle Pose (pose 2)

- Prediction of mean and std deviation of velocity and rotation in the local frame:
  - Velocity in x, y, z (m/s)
  - Angular velocity in roll, pitch, yaw (radians/s)
- **Total size:** 12 float32 values

#### 10. Recurrent State

- Recurrent state vector fed back into the GRU to provide temporal context.
- **Total size:** 512 float32 values

#### Output Index Mapping

| Index | Name                         | Index Range | Size |
|-------|------------------------------|-------------|------|
| 0     | Trajectory Planning          | 0 - 4954    | 4955 |
| 1     | Lane Line Detection          | 4955 - 5482 | 528  |
| 2     | Lane Line Probabilities      | 5483 - 5490 | 8    |
| 3     | Road Edges                   | 5491 - 5754 | 264  |
| 4     | Lead Vehicle Predictions     | 5755 - 5856 | 102  |
| 5     | Lead Vehicle Probabilities   | 5857 - 5859 | 3    |
| 6     | Desire State                 | 5860 - 5867 | 8    |
| 7     | Scene Metadata               | 5868 - 5947 | 80   |
| 8     | Vehicle Pose                 | 5948 - 5959 | 12   |
| 9     | Recurrent State              | 5960 - 6471 | 512  |

A **recurrent state vector** used as input to the **GRU (Gated Recurrent Unit)** to provide temporal context to the model.  
This vector has a size of **512 float32 values


## Acknowledgements
This work builds upon [comma.ai's openpilot](https://github.com/commaai/openpilot) and reuses parts of its codebase.