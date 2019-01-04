#graphics.py -- plotting functions for CamIO
#(c) James Coughlan, Smith-Kettlewell Eye Research Institute

import numpy as np
import cv2
from utilities import dict2list
import pygame

points_for_plotting_xyz_axes = np.array(((0.,0.,0.), (1.,0.,0.), (0.,1.,0.), (0.,0.,1.))) #p0, px, py, pz for plotting xyz axes
XYZ_AXES_LENGTH = 10. #length of axes to display, in cm

def plotXYZaxes(frameBGR, points_for_plotting_xyz_axes, XYZ_AXES_LENGTH, rvec, tvec, mtx, dist):
	#imgpoints, _ = cv2.projectPoints(points_for_plotting_xyz_axes*XYZ_AXES_LENGTH, np.squeeze(rvec), np.squeeze(tvec), mtx, dist)
	imgpoints, _ = cv2.projectPoints(points_for_plotting_xyz_axes*XYZ_AXES_LENGTH, rvec, tvec, mtx, dist)
	imgpoints2 = np.round(imgpoints).astype(int) #since plotting requires int coordinates!!!
	p0,px,py,pz = (imgpoints2[0][0][0],imgpoints2[0][0][1]),(imgpoints2[1][0][0],imgpoints2[1][0][1]),(imgpoints2[2][0][0],imgpoints2[2][0][1]),(imgpoints2[3][0][0],imgpoints2[3][0][1])		
	frameBGR = cv2.line(frameBGR, p0, px, (0,0,255),5)
	frameBGR = cv2.line(frameBGR, p0, py, (0,255,0),5)
	frameBGR = cv2.line(frameBGR, p0, pz, (255,100,0),5)
		
	return frameBGR
	
def plot_stylus(frameBGR, Xtip,Ytip,Ztip, rvec, tvec, mtx, dist):
	objp2 = np.array([[Xtip,Ytip,Ztip]])
	imgpoints, _ = cv2.projectPoints(objp2, np.squeeze(rvec), np.squeeze(tvec), mtx, dist)
	imgpoints2 = np.round(imgpoints).astype(int) #since plotting requires int coordinates!!!
	frameBGR = cv2.circle(frameBGR, (imgpoints2[0][0][0],imgpoints2[0][0][1]), 5, (0,0,255),5)
	
	return frameBGR

def plot_stylus_camera(frameBGR, Xtip,Ytip,Ztip, mtx, dist):
	objp2 = np.array([[Xtip,Ytip,Ztip]])
	imgpoints, _ = cv2.projectPoints(objp2, np.array([0.,0.,0.]), np.array([0.,0.,0.]), mtx, dist)
	imgpoints2 = np.round(imgpoints).astype(int) #since plotting requires int coordinates!!!
	frameBGR = cv2.circle(frameBGR, (imgpoints2[0][0][0],imgpoints2[0][0][1]), 5, (0,0,255),5)
	
	return frameBGR

'''
def plot_stylus_full(frameBGR, stylus_object, mtx, dist): #plot entire stylus object
	full_model3D = self.full_model3D[:,0,:]

	objp2 = np.array([[Xtip, Ytip, Ztip]])
	imgpoints, _ = cv2.projectPoints(objp2, np.array([0., 0., 0.]), np.array([0., 0., 0.]), mtx, dist)
	imgpoints2 = np.round(imgpoints).astype(int)  # since plotting requires int coordinates!!!
	frameBGR = cv2.circle(frameBGR, (imgpoints2[0][0][0], imgpoints2[0][0][1]), 5, (0, 0, 255), 5)

	return frameBGR
'''

def plot_corners(frameBGR, corners, ids): #for debugging visualization
	corners = np.round(corners).astype(int)
	if len(corners)>0:
		n, _, _ = np.shape(corners) #FIX THIS!
	else:
		n = 0
	for k in range(n):
		for c in range(4):
			frameBGR = cv2.circle(frameBGR, (corners[k,c,0], corners[k,c,1]), 5, (255,255,255),5)
		
	return frameBGR

def plot_hotspots(frameBGR, hotspots, current_hotspot, rvec, tvec, mtx, dist): #show all hotspot locations in green, except for current one in red
	#Note: if current_hotspot == 0, show all hotspots in green
	objp3 = np.array(dict2list(hotspots))
	imgpoints, _ = cv2.projectPoints(objp3, np.squeeze(rvec), np.squeeze(tvec), mtx, dist)
	imgpoints2 = np.round(imgpoints).astype(int) #since plotting requires int coordinates!!!
	for k in range(len(hotspots)):
		if (k+1) == current_hotspot:
			couleur = (0,0,255)
		else:
			couleur = (0,255,0)				
		frameBGR = cv2.circle(frameBGR, (imgpoints2[k][0][0],imgpoints2[k][0][1]), 5, couleur,5)	

	return frameBGR

def update_display(screen, frameBGR, decimation): #using pygame
	frameRGB = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2RGB)
	frameRGB = np.rot90(frameRGB) #I have no idea why this is necessary!
	frameRGB = frameRGB[::decimation,::decimation,:]
	frameRGB = pygame.surfarray.make_surface(frameRGB)
	frameRGB = pygame.transform.flip(frameRGB, True, False) #otherwise image is mirror reversed
	screen.blit(frameRGB, (0,0))
	pygame.display.update()
