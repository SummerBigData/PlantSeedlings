
import numpy as np
import matplotlib.pyplot as plt
#from skimage.filters import hessian



import dataPrep




r1 = np.array([[0.10434826, 0.955,      0.60422793, 0.89052632],
 [0.12349423, 0.94973684, 0.6060009,  0.87368421],
 [0.16895242, 0.93631579, 0.7122275,  0.84631579],
 [0.13717375, 0.94973684, 0.46811737, 0.89789474],
 [0.14123552, 0.94052632, 0.46590047, 0.88315789],
 [0.10951646, 0.95921053, 0.63217277, 0.88421053],
 [0.13320775, 0.94131579, 0.55916332, 0.86526316],
 [0.20511353, 0.92684211, 0.66527876, 0.87684211],
 [0.16979056, 0.92763158, 0.59078795, 0.85684211],
 [0.16070115, 0.92842105, 0.5185105, 0.87684211]])

r2 = np.array([[0.19100658, 0.92921053, 0.70295816, 0.87052632],
 [0.09542407, 0.95842105, 0.61775911, 0.8831579 ],
 [0.32164158, 0.87710526, 0.73913095, 0.81157895],
 [0.13731588, 0.94631579, 0.49013395, 0.88736842],
 [0.16075859, 0.93394737, 0.50663857, 0.88105263],
 [0.20572623, 0.93315789, 0.83905705, 0.85684211],
 [0.20158538, 0.93157895, 0.59358081, 0.88210526],
 [0.20711252, 0.92526316, 0.66721518, 0.86947368],
 [0.27176535, 0.91052632, 0.74036799, 0.86842105],
 [0.14572224, 0.93236842, 0.48244416, 0.87789474]])

print np.mean(r1[:,3])

print np.mean(r2[:,3])



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

'''

numSplits = 3


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
	
	imgPlot[i*4] = imgs[i]
	imgPlot[i*4+1] = maskRGB
	imgPlot[i*4+2] = stamp
	imgPlot[i*4+3] = stampBW
	
#PlotImgs(imgPlot, numImgs, numSplits+1)

np.save('data/trainImgsRes'+str(dim)+'stampBW'+str(dim), stampBWall)

'''


