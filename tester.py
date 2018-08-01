
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import hessian
import cv2


import dataPrep






def PlotImgs(imgs, numImgs):
	fig, axes = plt.subplots(nrows=numImgs, ncols=4)
	ax = axes.ravel()
	for i in range(imgs.shape[0]):
		ax[i].imshow(imgs[i])
		#ax[i].set_title(label[labels[i]])
	#plt.tight_layout()
	plt.show()

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


imgs, labels = dataPrep.getTrainDat(100)


'''
imgsDC = np.zeros((imgs.shape))
imgsHess = np.zeros((4, 100, 100, 3))


for i in range(4):
	for x in range(100):
		for y in range(100):
			R = imgs[i, x, y, 0]
			G = imgs[i, x, y, 1]
			B = imgs[i, x, y, 2]
			RGB = np.array([R, G, B])
			avg = np.mean(RGB) - 0.5
			imgs[i, x, y, :] -= avg

	imgsHess[i, :, :, 0] = hessian(imgs[i, :, :, 1], beta1 = .1)



imgPlot = np.zeros((5, 100, 100, 3))
imgPlot[0, :, :, 0] = imgs[0, :, :, 0]
imgPlot[1, :, :, 1] = imgs[0, :, :, 1]
imgPlot[2, :, :, 2] = imgs[0, :, :, 2]
imgPlot[3, :, :, :] = imgs[0]
imgPlot[4, :, :, :] = imgsHess[0]

PlotImgs(imgPlot, 5)
'''




imgPlot = np.zeros((40, 100, 100, 3))

for i in range(20):
	fmatted = (imgs[i]*255.0).astype(np.uint8)
	mask = create_mask_for_plant( fmatted )
	mask = mask.reshape((100, 100, 1))
	maskRGB = np.concatenate(( mask, mask, mask), axis=2) / 255.0
	print imgs[i].min(), imgs[i].max()
	print mask.min(), mask.max()
	print ' '
	imgPlot[i*2] = imgs[i]
	imgPlot[i*2+1] = maskRGB


PlotImgs(imgPlot, 10)





