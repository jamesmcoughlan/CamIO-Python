#UI.py
#(c) James Coughlan, Smith-Kettlewell Eye Research Institute

import numpy as np
import cv2
import time
from copy import deepcopy
#from utilities import load_object, save_object, dist2hotspots3D, dict2list, get_board_pose, convert_coors_from_camera_to_marker, convert_coors_from_marker_to_camera
from stylus import Stylus, Markers
from sounds import Sounds, AMBIENT_PERIOD
from camera import Camera, CAMERAS, decimations
from geometry import get_ground_rvec_tvec2, assemble_transformation, convert4x4_to_rvec_tvec, append_1, get_specific_rvec_tvec, determine_poses
from parameters import *

#############################################################

def scan_ground_plane_marker(corners, ids, camera_object, sound_object):
	plane_pose, Tac = None, None
	if ids is not None:
		if ground_plane_marker_ID in ids:
			rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, ground_plane_marker_length, camera_object.mtx, camera_object.dist)
			plane_pose = get_specific_rvec_tvec(rvecs, tvecs, ground_plane_marker_ID, ids)
			Tmc = assemble_transformation(plane_pose[0], plane_pose[1])
			Tac = Tmc + 0. #in this application they are identical
			pose_known = True
			print('Estimated ground plane.')
			#sound_object.play_mp3('sounds/ground plane captured.mp3')
			
	return plane_pose, Tac

def estimate_ground_plane_from_two_stylus_scans(stylus_info_at_location_1, stylus_info_at_location_2, sound_object):
	if (stylus_info_at_location_1 is not None) and (stylus_info_at_location_2 is not None):
		plane_pose = get_ground_rvec_tvec2(stylus_info_at_location_1, stylus_info_at_location_2)
		Tmc = assemble_transformation(plane_pose[0], plane_pose[1])
		Tac = Tmc + 0. #in this application they are identical
		pose_known = True
		print('Estimated ground plane.')
		#TO DO: add sound
		#sound_object.play_mp3('sounds/pose complete.mp3')

	return plane_pose, Tac

def save_stylus_info(stylus_object, sound_object):
	stylus_info_at_location = deepcopy(stylus_object)
	print('Stylus position captured.')
	#TO DO: add sound
	return stylus_info_at_location

def estimate_pose(stylus_info_at_location_a, stylus_info_at_location_b, plane_pose, anchor_1_XYZ, anchor_2_XYZ, sound_object):
	pose_known = False
	pose = None
	Tca = None
	if (stylus_info_at_location_a is not None) and (stylus_info_at_location_b is not None) and (plane_pose is not None):
		anchor_a = stylus_info_at_location_a.tip_XYZ
		anchor_b = stylus_info_at_location_b.tip_XYZ
		Tmc, Tcm, Tma, Tca, Tac = determine_poses(plane_pose, anchor_a, anchor_b, anchor_1_XYZ, anchor_2_XYZ)
		pose_known = True
		rvec, tvec = convert4x4_to_rvec_tvec(Tac)
		pose = rvec, tvec
		print('Estimated pose.')
		#TO DO: add sound
	else:
		print("Can't estimate pose.")
	return pose_known, pose, Tca

def estimate_stylus_location_in_annotation_coors(stylus_location_XYZ_raw, Tca, sound_object):
	stylus_location_XYZ = Tca @ append_1(stylus_location_XYZ_raw)
	stylus_location_XYZ = stylus_location_XYZ[0:3]

	return stylus_location_XYZ

def take_action(obs_smoothed, obs_smoothed_old, sound_object):
	'''Issue feedback if appropriate.'''

	if obs_smoothed_old != obs_smoothed: #transition to new state
		print('Transitioned to new hotspot:', obs_smoothed)
		sound_object.halt_TTS()
		if obs_smoothed > 0:
			sound_object.play_hotspot(obs_smoothed)

	return obs_smoothed + 0, obs_smoothed + 0

def quit_video(cap):
	print('Quit program.')
	cap.release()
	cv2.destroyAllWindows()
	return
