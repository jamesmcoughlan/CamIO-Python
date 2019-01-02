#stylus.py
#(c) James Coughlan, Smith-Kettlewell Eye Research Institute

#code to support stylus, including markers printed on it

import numpy as np
import cv2

def detectMarkers_clean(gray, dictionary, parameters): #call aruco.detectMarkers but clean up the returned corners and ids
	#corners: multi-array of dimension n x 4 x 2
	#ids: multi-array of dimension n
	corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary=dictionary, parameters=parameters)
	if len(corners)>0:
		corners = np.array(corners)
		corners = corners[:,0,:,:] #get rid of the useless dimension
		ids = ids[:,0] #get rid of the useless dimension
	return corners, ids

def arrange_corresponding_points(objbases, corners, ids): #create corresponding lists of (detected) 2D image points and 3D model points
	imgpoints, objpoints = [], [] #create corresponding lists of (detected) 2D image points and 3D model points
	for this_id in ids:
		if this_id in list(objbases.keys()): #otherwise irrelevant and should be ignored
			inds = np.where(ids==this_id)
			that_ind = inds[0][0] #index corresponding to probe marker with id
			corners4 = corners[that_ind,:,:]
			imgpoints.append(corners4)
			objpoints.append(objbases[this_id])
			
			center = np.round(np.average(corners4, axis=0)).astype(int)

	if len(imgpoints)>0:
		imgpoints2 = np.array(imgpoints)
		imgpoints3 = np.ascontiguousarray(imgpoints2[:,:,None]).reshape((len(imgpoints)*4,2,1)) #want contiguous n x 2 x 1
		objpoints2 = np.array(objpoints)
		objpoints3 = np.ascontiguousarray(objpoints2[:,:,None]).reshape((len(imgpoints)*4,3,1)) #promote to n x 3 x 1 dimensions, then make sure it's contiguous
		return True, objpoints3, imgpoints3
	else:
		return False, None, None
	
def projection_error(imgpoints, objpoints, rvec, tvec, mtx, dist): #n x 2 x 1, n x 1 x 3 respectively
	n, _, _ = imgpoints.shape
	reprojected, _ = cv2.projectPoints(objpoints, np.squeeze(rvec), np.squeeze(tvec), mtx, dist)
	errors = [cv2.norm(reprojected[k,0,:] - imgpoints[k,:,0]) for k in range(n)]
	#print('errors:', errors)
	return errors

	
def stylus_half_inch_dowel(offset, objbases, corners, ids, mtx, dist):
	#1/2" square dowel stylus, variable length

	condition, objpoints3, imgpoints3 = arrange_corresponding_points(objbases, corners, ids)
	if condition:
		ret, rvec, tvec = cv2.solvePnP(objpoints3, imgpoints3, mtx, dist)
		errors = projection_error(imgpoints3, objpoints3, rvec, tvec, mtx, dist)
		E = np.max(errors)
		if E<10 and len(errors)>=8: #if reprojection error is not too bad, and there are at least two markers detected
			R,_ = cv2.Rodrigues(rvec)
			tip = (R@offset) + np.squeeze(tvec) #XYZ tip location in camera coordinates
			return True, tip, rvec, tvec
		else:
			return False, 0*offset, 0*offset, 0*offset
	else:
		return False, 0*offset, 0*offset, 0*offset

def stylus_1_inch_dowel(offset, objbases, corners, ids, mtx, dist):
	#1" square dowel stylus, variable length

	condition, objpoints3, imgpoints3 = arrange_corresponding_points(objbases, corners, ids)
	if condition:
		ret, rvec, tvec = cv2.solvePnP(objpoints3, imgpoints3, mtx, dist)
		errors = projection_error(imgpoints3, objpoints3, rvec, tvec, mtx, dist)
		E = np.max(errors)
		if E<10 and len(errors)>=8: #if reprojection error is not too bad, and there are at least two markers detected
			R,_ = cv2.Rodrigues(rvec)
			tip = (R@offset) + np.squeeze(tvec) #XYZ tip location in camera coordinates
			return True, tip, rvec, tvec
		else:
			return False, 0*offset, 0*offset, 0*offset
	else:
		return False, 0*offset, 0*offset, 0*offset

def stylus_2_inch_dowel(offset, objbases, corners, ids, mtx, dist):
	#2" square dowel stylus, variable length

	condition, objpoints3, imgpoints3 = arrange_corresponding_points(objbases, corners, ids)
	if condition:
		ret, rvec, tvec = cv2.solvePnP(objpoints3, imgpoints3, mtx, dist)
		errors = projection_error(imgpoints3, objpoints3, rvec, tvec, mtx, dist)
		E = np.max(errors)
		if E<10 and len(errors)>=8: #if reprojection error is not too bad, and there are at least two markers detected
			R,_ = cv2.Rodrigues(rvec)
			tip = (R@offset) + np.squeeze(tvec) #XYZ tip location in camera coordinates
			return True, tip, rvec, tvec
		else:
			return False, 0*offset, 0*offset, 0*offset
	else:
		return False, 0*offset, 0*offset, 0*offset

def stylus_2_inch_box(offset, objbases, corners, ids, mtx, dist):
	#stylus_2_inch_box, variable length

	condition, objpoints3, imgpoints3 = arrange_corresponding_points(objbases, corners, ids)
	if condition:
		ret, rvec, tvec = cv2.solvePnP(objpoints3, imgpoints3, mtx, dist)
		errors = projection_error(imgpoints3, objpoints3, rvec, tvec, mtx, dist)
		E = np.max(errors)
		#print('errors:',errors)
		if E<10 and len(errors)>=4: #if reprojection error is not too bad, and there is at least one marker detected
			R,_ = cv2.Rodrigues(rvec)
			tip = (R@offset) + np.squeeze(tvec) #XYZ tip location in camera coordinates
			return True, tip, rvec, tvec
		else:
			return False, 0*offset, 0*offset, 0*offset
	else:
		return False, 0*offset, 0*offset, 0*offset
		
#########

class Markers:
	def __init__(self):
		#probe markers for stylus:
		self.arucodict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
		self.arucodetectparam = cv2.aruco.DetectorParameters_create()
		#self.arucodetectparam.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX #this helps with aruco markers but not charuco board
		self.arucodetectparam.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR #this helps with aruco markers but not charuco board
		self.arucodetectparam.cornerRefinementWinSize = 3 #3
		#arucodetectparam.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE #Debugging!
		#arucodetectparam.cornerRefinementMinAccuracy = 0.05 #debugging
		#arucodetectparam.cornerRefinementMaxIterations = 50

		if 1: #empirically, these settings seem to improve marker localization
			self.arucodetectparam.adaptiveThreshWinSizeMin = 33  #default 3 but maybe 33 is good
			self.arucodetectparam.adaptiveThreshWinSizeMax = 53  #default 23 but maybe 53 is good
			#self.arucodetectparam.adaptiveThreshWinSizeStep = 10  #default 10 and this is good

class Stylus:
	def __init__(self, which, length, mtx, dist): 
		self.which = which #which specifies the type of stylus
		self.mtx = mtx
		self.dist = dist
		self.PROBE_LENGTH = length
							
		if which == 'stylus_half_inch_dowel': #origin is in center marker on top ("eraser") of stylus
			self.func = stylus_half_inch_dowel
			
			b = 1.27 #width in cm of white square (bounded by lines on all four sides)
			a = 500/800*b #width in cm of Aruco marker
			margin = (b-a)/2 #margin in cm between marker and edge of paper

			offset = np.array([0.,0.,-self.PROBE_LENGTH])
			
			objbases = {}
			objbases[0] = np.array([[-a/2,a/2,0.],[a/2,a/2,0.],[a/2,-a/2,0.],[-a/2,-a/2,0.]]) #marker id 0
			
			objbases[1] = np.array([[-a/2,-b/2,-margin],[a/2,-b/2,-margin],[a/2,-b/2,-b+margin],[-a/2,-b/2,-b+margin]]) #marker id 1
			for k in range(2,12):
				objbases[k] = objbases[1] + (k-1)*np.array([0.,0.,-b])

			objbases[21] = np.array([[b/2,-a/2,-margin],[b/2,a/2,-margin],[b/2,a/2,-b+margin],[b/2,-a/2,-b+margin]]) #marker id 21
			for k in range(22,32):
				objbases[k] = objbases[21] + (k-21)*np.array([0.,0.,-b])	

			objbases[41] = np.array([[a/2,b/2,-margin],[-a/2,b/2,-margin],[-a/2,b/2,-b+margin],[a/2,b/2,-b+margin]]) #marker id 41, problem here?
			for k in range(42,52):
				objbases[k] = objbases[41] + (k-41)*np.array([0.,0.,-b])	

			objbases[61] = np.array([[-b/2,a/2,-margin],[-b/2,-a/2,-margin],[-b/2,-a/2,-b+margin],[-b/2,a/2,-b+margin]]) #marker id 61
			for k in range(62,72):
				objbases[k] = objbases[61] + (k-61)*np.array([0.,0.,-b])

			n = len(objbases)
			full_model3D = []
			for id in objbases:
				full_model3D.append(objbases[id][0,:])
				full_model3D.append(objbases[id][1,:])
				full_model3D.append(objbases[id][2,:])
				full_model3D.append(objbases[id][3,:])
			full_model3D = np.array(full_model3D)
			full_model3D = full_model3D[:,None,:]		
			
			self.offset = offset
			self.full_model3D = full_model3D
			self.objbases = objbases
			self.a = a
			self.b = b
			self.margin = margin

		if which == 'stylus_1_inch_dowel':  # origin is in center marker on top ("eraser") of stylus
			self.func = stylus_half_inch_dowel

			b = 2.54  # width in cm of white square (bounded by lines on all four sides)
			a = 500 / 800 * b  # width in cm of Aruco marker
			margin = (b - a) / 2  # margin in cm between marker and edge of paper

			offset = np.array([0., 0., -self.PROBE_LENGTH])

			objbases = {}
			objbases[0] = np.array(
				[[-a / 2, a / 2, 0.], [a / 2, a / 2, 0.], [a / 2, -a / 2, 0.], [-a / 2, -a / 2, 0.]])  # marker id 0

			objbases[1] = np.array([[-a / 2, -b / 2, -margin], [a / 2, -b / 2, -margin], [a / 2, -b / 2, -b + margin],
									[-a / 2, -b / 2, -b + margin]])  # marker id 1
			for k in range(2, 12):
				objbases[k] = objbases[1] + (k - 1) * np.array([0., 0., -b])

			objbases[21] = np.array([[b / 2, -a / 2, -margin], [b / 2, a / 2, -margin], [b / 2, a / 2, -b + margin],
									 [b / 2, -a / 2, -b + margin]])  # marker id 21
			for k in range(22, 32):
				objbases[k] = objbases[21] + (k - 21) * np.array([0., 0., -b])

			objbases[41] = np.array([[a / 2, b / 2, -margin], [-a / 2, b / 2, -margin], [-a / 2, b / 2, -b + margin],
									 [a / 2, b / 2, -b + margin]])  # marker id 41, problem here?
			for k in range(42, 52):
				objbases[k] = objbases[41] + (k - 41) * np.array([0., 0., -b])

			objbases[61] = np.array([[-b / 2, a / 2, -margin], [-b / 2, -a / 2, -margin], [-b / 2, -a / 2, -b + margin],
									 [-b / 2, a / 2, -b + margin]])  # marker id 61
			for k in range(62, 72):
				objbases[k] = objbases[61] + (k - 61) * np.array([0., 0., -b])

			n = len(objbases)
			full_model3D = []
			for id in objbases:
				full_model3D.append(objbases[id][0, :])
				full_model3D.append(objbases[id][1, :])
				full_model3D.append(objbases[id][2, :])
				full_model3D.append(objbases[id][3, :])
			full_model3D = np.array(full_model3D)
			full_model3D = full_model3D[:, None, :]

			self.offset = offset
			self.full_model3D = full_model3D
			self.objbases = objbases
			self.a = a
			self.b = b
			self.margin = margin

		if which == 'stylus_2_inch_dowel':  # origin is in center marker on top ("eraser") of stylus
			self.func = stylus_half_inch_dowel

			b = 2*2.54  # width in cm of white square (bounded by lines on all four sides)
			a = 500 / 800 * b  # width in cm of Aruco marker
			margin = (b - a) / 2  # margin in cm between marker and edge of paper

			offset = np.array([0., 0., -self.PROBE_LENGTH])

			objbases = {}
			objbases[0] = np.array(
				[[-a / 2, a / 2, 0.], [a / 2, a / 2, 0.], [a / 2, -a / 2, 0.], [-a / 2, -a / 2, 0.]])  # marker id 0

			objbases[1] = np.array([[-a / 2, -b / 2, -margin], [a / 2, -b / 2, -margin], [a / 2, -b / 2, -b + margin],
									[-a / 2, -b / 2, -b + margin]])  # marker id 1
			for k in range(2, 12):
				objbases[k] = objbases[1] + (k - 1) * np.array([0., 0., -b])

			objbases[21] = np.array([[b / 2, -a / 2, -margin], [b / 2, a / 2, -margin], [b / 2, a / 2, -b + margin],
									 [b / 2, -a / 2, -b + margin]])  # marker id 21
			for k in range(22, 32):
				objbases[k] = objbases[21] + (k - 21) * np.array([0., 0., -b])

			objbases[41] = np.array([[a / 2, b / 2, -margin], [-a / 2, b / 2, -margin], [-a / 2, b / 2, -b + margin],
									 [a / 2, b / 2, -b + margin]])  # marker id 41, problem here?
			for k in range(42, 52):
				objbases[k] = objbases[41] + (k - 41) * np.array([0., 0., -b])

			objbases[61] = np.array([[-b / 2, a / 2, -margin], [-b / 2, -a / 2, -margin], [-b / 2, -a / 2, -b + margin],
									 [-b / 2, a / 2, -b + margin]])  # marker id 61
			for k in range(62, 72):
				objbases[k] = objbases[61] + (k - 61) * np.array([0., 0., -b])

			n = len(objbases)
			full_model3D = []
			for id in objbases:
				full_model3D.append(objbases[id][0, :])
				full_model3D.append(objbases[id][1, :])
				full_model3D.append(objbases[id][2, :])
				full_model3D.append(objbases[id][3, :])
			full_model3D = np.array(full_model3D)
			full_model3D = full_model3D[:, None, :]

			self.offset = offset
			self.full_model3D = full_model3D
			self.objbases = objbases
			self.a = a
			self.b = b
			self.margin = margin

		if which == 'stylus_2_inch_box': #origin is in center marker on top ("eraser") of stylus
			self.func = stylus_2_inch_box
			
			b = 5.08 #width in cm of white square (bounded by lines on all four sides)
			a = 500/800*b #width in cm of Aruco marker
			margin = (b-a)/2 #margin in cm between marker and edge of paper

			offset = np.array([0.,0.,-self.PROBE_LENGTH])
			
			objbases = {}
			objbases[0] = np.array([[-a/2,a/2,0.],[a/2,a/2,0.],[a/2,-a/2,0.],[-a/2,-a/2,0.]]) #marker id 0
			objbases[1] = np.array([[-a/2,-b/2,-margin],[a/2,-b/2,-margin],[a/2,-b/2,-b+margin],[-a/2,-b/2,-b+margin]]) #marker id 1
			objbases[2] = np.array([[b/2,-a/2,-margin],[b/2,a/2,-margin],[b/2,a/2,-b+margin],[b/2,-a/2,-b+margin]]) #marker id 2
			objbases[3] = np.array([[a/2,b/2,-margin],[-a/2,b/2,-margin],[-a/2,b/2,-b+margin],[a/2,b/2,-b+margin]]) #marker id 3
			objbases[4] = np.array([[-b/2,a/2,-margin],[-b/2,-a/2,-margin],[-b/2,-a/2,-b+margin],[-b/2,a/2,-b+margin]]) #marker id 4

			n = len(objbases)
			full_model3D = []
			for id in objbases:
				full_model3D.append(objbases[id][0,:])
				full_model3D.append(objbases[id][1,:])
				full_model3D.append(objbases[id][2,:])
				full_model3D.append(objbases[id][3,:])
			full_model3D = np.array(full_model3D)
			full_model3D = full_model3D[:,None,:]		
			
			self.offset = offset
			self.full_model3D = full_model3D
			self.objbases = objbases
			self.a = a
			self.b = b
			self.margin = margin

	def apply_image(self, corners, ids):  # apply image info, calculate results and save
		self.visible = False
		self.rvec = None
		self.tvec = None
		self.tip_XYZ = None #np.array([0., 0., 0.])
		self.corners = corners
		self.ids = ids
		if len(corners) > 0:
			visible, tip_XYZ, rvec, tvec = self.func(self.offset, self.objbases, corners, ids, self.mtx, self.dist)
			self.rvec = np.squeeze(rvec)
			self.tvec = np.squeeze(tvec)
			self.visible = visible
			self.tip_XYZ = tip_XYZ
