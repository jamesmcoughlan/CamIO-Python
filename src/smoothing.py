#smoothing.py
#(c) James Coughlan, Smith-Kettlewell Eye Research Institute

#Code to implement smoothing of stylus location.
#Current version smoothes the quantized hotspot location.

import numpy as np
#import cv2
from collections import deque # https://www.idiotinside.com/2015/03/01/python-lists-as-fifo-lifo-queues-using-deque-collections/
from collections import Counter #https://www.reddit.com/r/learnprogramming/comments/78vuur/python_histogram_using_list_comprehension/
from parameters import DT

class Smoothing:
	def __init__(self):
		self.dq_time = deque()  #FIFO queue
		self.dq_state = deque() #FIFO queue (need a separate one since the .most_common() method only looks at obs, not time)

	def add_observation(self, obs, timestamp):
		self.dq_time.append(timestamp)
		self.dq_state.append(obs)  # append to right
		if len(self.dq_time) > 0:
			if timestamp - self.dq_time[0] > DT:  # kill elements of queue older than time DT:
				self.dq_time.popleft()
				self.dq_state.popleft()

		# smooth observations over time
		counts = Counter(self.dq_state)
		L = counts.most_common()
		obs_smoothed = 0  # by default
		L_len = len(self.dq_state)
		if L_len > 0:
			majority = (L_len + (L_len % 2)) / 2  # e.g., 3 if L_len = 5, or 2 if L_len = 4
			if L[0][1] >= majority:  # go with the mode if it's popular enough
				obs_smoothed = L[0][0]  # most common observation in queue

		return obs_smoothed


