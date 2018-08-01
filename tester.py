
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import hessian



import dataPrep






def PlotImgs(imgs, nrows, ncols):
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
	ax = axes.ravel()
	for i in range(imgs.shape[0]):
		ax[i].imshow(imgs[i])
		ax[i].axes.get_yaxis().set_visible(False)
		ax[i].axes.get_xaxis().set_visible(False)
		#ax[i].set_title(label[labels[i]])
	#plt.tight_layout()
	plt.show()








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


dim = 400
numSplits = 3

imgs, labels = dataPrep.getTrainDat(dim)
numImgs = imgs.shape[0]

#imgPlot = np.zeros(( (numSplits+1)*numImgs, dim, dim, 3))
stampBWall = np.zeros((imgs.shape[0], dim, dim))

for i in range(numImgs):
	if i%100 ==0:
		print 'image', i
		print ''
	mask = dataPrep.getPlantMask(imgs[i], 36)
	mask = mask.reshape((dim, dim, 1))
	maskRGB = np.concatenate((mask, mask, mask), axis = 2)
	
	stamp = np.multiply(maskRGB, imgs[i])
	stampBW = np.mean(stamp, axis=2)
	stampBW = stampBW.reshape((dim, dim, 1))
	stampBW = np.concatenate((stampBW, stampBW, stampBW), axis = 2)
	stampBW, _, _ = dataPrep.Norm(stampBW, 0, 1)
	stampBWall[i] = stampBW[:, :, 0]
	'''
	imgPlot[i*4] = imgs[i]
	imgPlot[i*4+1] = maskRGB
	imgPlot[i*4+2] = stamp
	imgPlot[i*4+3] = stampBW
	'''
#PlotImgs(imgPlot, numImgs, numSplits+1)

np.save('data/trainImgsRes'+str(dim)+'stampBW'+str(dim), stampBWall)





