import numpy as np

rvec = np.deg2rad(np.array([0., 0.0, 0.0]))
tvec = np.array([0.2098, 0.0,-1.585])

FULL_FRAME_SIZE = (1160, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0
eon_intrinsics = np.array([
  [FOCAL,   0.,   W/2.],
  [  0.,  FOCAL,  H/2.],
  [  0.,    0.,     1.]])

dist_coeff = np.array([ -0.48955719  ,0.00689698 ,-0.01130375 ,-0.0245049 ,  1.00655564 ])

leon_dcam_intrinsics = np.array([
  [650,   0,   816//2],
  [  0,  650,  612//2],
  [  0,    0,     1]])

eon_dcam_intrinsics = np.array([
  [860,   0,   1152//2],
  [  0,  860,  864//2],
  [  0,    0,     1]])

CAMARA_HEIGHT = 1.691
CAR_WIDTH = 0.8