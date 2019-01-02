#constants.py
#(c) James Coughlan, Smith-Kettlewell Eye Research Institute

CAMERAS = {'laptop-VGA':('../res/calib-laptop/',(480,640)), 'laptop-720p':('../res/calib-laptop 720p/', (720,1280)), 'BRIO-VGA':('../res/calib-Logitech BRIO VGA/',(480,640)), 
'BRIO-1080p':('../res/calib-Logitech BRIO 1080p/',(1080,1920)), 'BRIO-max':('../res/calib-Logitech BRIO max/', (2160, 4096)), 'HUE-720p':('../res/calib-HUE 720p/', (720,1280)),
'C920':('../res/calib-Logitech C920/',(1080,1920))}

decimations = {'laptop-VGA':1, 'laptop-720p':1, 'BRIO-VGA':1, 'BRIO-1080p':2, 'BRIO-max':4, 'HUE-720p':1, 'C920':2} #for display purposes