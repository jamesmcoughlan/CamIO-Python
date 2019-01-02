#camera.py
#(c) James Coughlan, Smith-Kettlewell Eye Research Institute

#code and constants to support camera

import numpy as np

#constants:
CAMERAS = {'laptop-VGA':('calib-laptop/',(480,640)), 'laptop-720p':('calib-laptop 720p/', (720,1280)), 'BRIO-VGA':('calib-Logitech BRIO VGA/',(480,640)), 
'BRIO-1080p':('calib-Logitech BRIO 1080p/',(1080,1920)), 'BRIO-max':('calib-Logitech BRIO max/', (2160, 4096)), 'HUE-720p':('calib-HUE 720p/', (720,1280)),
'C920':('calib-Logitech C920/',(1080,1920))}

decimations = {'laptop-VGA':1, 'laptop-720p':1, 'BRIO-VGA':1, 'BRIO-1080p':2, 'BRIO-max':4, 'HUE-720p':1, 'C920':2} #for display purposes

class Camera:
	def __init__(self, camera): 
		self.camera_calibration_path = camera[0] #which camera
		self.h = camera[1][0]
		self.w = camera[1][1]
		self.mtx, self.dist = np.loadtxt(self.camera_calibration_path+'mtx.txt'), np.loadtxt(self.camera_calibration_path+'dist.txt')
