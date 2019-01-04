import pygame
from pygame.locals import *
import cv2
import numpy as np
import sys

camera = cv2.VideoCapture(0)
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT,1080) #set camera image height
#camera.set(cv2.CAP_PROP_FRAME_WIDTH,1920) #set camera image width
camera.set(cv2.CAP_PROP_FOCUS,0) #set focus

pygame.init()
#pygame.display.set_caption("OpenCV camera stream on Pygame")
screen = pygame.display.set_mode([640,480])
#screen = pygame.display.set_mode([1080,1920])

try:
	while True:
		ret, frame = camera.read()
		keyval = cv2.waitKey(30) & 0xFF
		print('keyval:',keyval)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#frame = cv2.circle(frame, (10, 150), 5, (255,255,0),5) #test graphics --> success!
		
		frame = np.rot90(frame) #I have no idea why this is necessary!
		frame = pygame.surfarray.make_surface(frame)
		frame = pygame.transform.flip(frame, True, False) #otherwise image is mirror reversed
		screen.blit(frame, (0,0))
		pygame.display.update()

		if keyval == ord('0'):
			print('0!!!!!!!!!!!!!!!!!!!!!')
		for event in pygame.event.get():
			#if event.type == KEYDOWN:
				#sys.exit(0)
			if event.type == KEYDOWN:
				sys.exit(0)

except (KeyboardInterrupt,SystemExit):
	pygame.quit()
	cv2.destroyAllWindows()
