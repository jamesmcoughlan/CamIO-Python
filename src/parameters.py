#parameters.py
#(c) James Coughlan, Smith-Kettlewell Eye Research Institute

#All user parameters needed for CamIO

from constants import *
import numpy as np

#parameters for user to specify:

#which_camera = 'BRIO-1080p'
#which_camera = 'HUE-720p'
which_camera = 'C920'
#which_camera = 'laptop-720p'
int_or_ext = 0 #internal webcam or external USB webcam (doesn't work reliably, but when an external webcam is plugged in, 1 usually means internal and 0 external)

#which_stylus = 'stylus_half_inch_dowel'; stylus_length = 15.0 #length from top to tip in cm
#which_stylus = 'stylus_1_inch_dowel'; stylus_length = 12.8 #length from top to tip in cm
#which_stylus = 'stylus_2_inch_box'; stylus_length = 15.3 + 0*6.2 # 15.3 #length from top to tip in cm
which_stylus = 'stylus_2_inch_dowel'; stylus_length = 22.2 #length from top to tip in cm

ground_plane_marker_ID = 249
ground_plane_marker_length = 5.0 #cm

#object_path, object_fname = 'objects/house/', 'output.xlsx'
#object_path, object_fname = 'objects/Southwest/', 'output.xlsx'
#object_path, object_fname = 'objects/cell model/', 'output.xlsx'
object_path, object_fname = '../res/objects/Southwest/', 'output.xlsx'

anchor_1_ind, anchor_2_ind = 1, 2 #by default, it's always the first two hotspots, respectively

DIST_THRESH = 3. #distance threshold, in same units as spreadsheet (cm), to be considered at a hotspot
K_DIST = 1.2 #2nd closest hotspot distance must be at least this factor times the closest hotspot, in order for closest hotspot to be triggered
DT = 0.75 #time interval (sec.) over which smoothing is performed
