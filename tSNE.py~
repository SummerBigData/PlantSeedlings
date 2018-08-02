import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import dataPrep





'''
ind = []

imgs, labels = dataPrep.getTrainDat(dim)
numImgs = imgs.shape[0]

#imgPlot = np.zeros(( (numSplits+1)*numImgs, dim, dim, 3))
stampBWall = np.zeros((imgs.shape[0], dim, dim))

for i in range(numImgs):
	if i%100 ==0:
		print 'image', i
		print ''
	mask = dataPrep.getPlantMask(imgs[i], 36)
	
	if mask.max() < 1e-8:
		ind.append(i)
		
	mask = mask.reshape((dim, dim, 1))
	maskRGB = np.concatenate((mask, mask, mask), axis = 2).astype(int)
 
	stamp = np.multiply(maskRGB, imgs[i])

	stampBW = np.mean(stamp, axis=2)
	#stampBW = stampBW.reshape((dim, dim, 1))
	#stampBW = np.concatenate((stampBW, stampBW, stampBW), axis = 2)
	
	stampBWall[i] = stampBW

	#imgPlot[i*4] = imgs[i]
	#imgPlot[i*4+1] = maskRGB
	#imgPlot[i*4+2] = stamp
	#imgPlot[i*4+3] = stampBW

stampBWall, _, _ = dataPrep.Norm(stampBWall, 0, 1)
#PlotImgs(imgPlot, numImgs, numSplits+1)

print ind

np.save('data/trainImgsRes'+str(dim)+'stampBW'+str(dim), stampBWall)
'''


'''


dim = 100

#

datStr = 'data/trainImgsRes400stampBW400.npy'

stamp = np.load(datStr)
stamp = dataPrep.resizeDat(datStr, dim)

imgs, labels = dataPrep.getTrainDat(dim)

avgBlue = np.mean(imgs[:,:,:,2])
print 'avg blue', avgBlue 
stdBlue = np.std(imgs[:,:,:,2])
print 'stdev blue', stdBlue
avg = np.mean(imgs)
print 'avg', avg
std = np.std(imgs)
print 'stdev', std
print imgs.shape, stamp.shape


#print str(stamp[108, 0, 0])  == str(np.nan)

#plotImgs = np.zeros((len(ind)*3, dim, dim, 3))
#dataPrep.PlotImgs(stamp[108:], 20, 2)


for i in range(imgs.shape[0]):
	if str(stamp[i, 0, 0] ) == 'nan':
		newImg = imgs[i]
		newMean, newStd = np.mean(newImg), np.std(newImg)
		adjMean, adjStd = 
		newImg 
		newImg[:,:,2],_,_ = dataPrep.Norm(newImg[:,:,2], -stdBlue*1.5, stdBlue*1.5)
		newImg[:,:,2] += avgBlue
		
		print i, 'min', newImg[:, :, 2].min(),'max', newImg[:, :, 2].max()
		imgs[i] = newImg 
		print newImg.shape, imgs[i].shape
		dataPrep.PlotImgs( imgs[i:], 2, 1)
		stop
		mask = dataPrep.getPlantMask(newImg, 36)
		print i, 'min', mask[:, :].min(),'max', mask[:, :].max()
	
		mask = mask.reshape((dim, dim, 1))
		maskRGB = np.concatenate((mask, mask, mask), axis = 2).astype(int)
	 	
		stampBW = np.multiply(maskRGB, imgs[i])

		stampBW = np.mean(stampBW, axis=2)
		#stampBW = stampBW.reshape((dim, dim, 1))
		#stampBW = np.concatenate((stampBW, stampBW, stampBW), axis = 2)
		
		stampBWall[i] = stampBW
		print i, 'min', stampBWall[:, :].min(),'max', stampBWall[:, :].max()

		print ' '

np.save('data/FixedImgs/'+ 'trainImgsResized'+str(dim)+'.npy', imgs)
np.save('data/'+'trainImgsRes400stampBW'+str(dim)+'Fixed.npy', stampBWall)
'''



datStr = 'data/trainImgsRes400stampBW400.npy'
ind = np.array([108, 155, 219, 222, 1436, 1564, 2250, 2302, 2346, 2358, 2382, 2398, 2414, 2584, 
		2624, 2631, 2644, 2650, 2669, 2692, 2723, 2727, 2752, 2824, 2825, 2850, 2876, 
		2893])

imgs = dataPrep.resizeDat(datStr, 100)
_, numlabels = dataPrep.getTrainDat(100)

print ' '
#print imgs.min(), imgs.max(),imgs.shape

imgsToUse = np.zeros((imgs.shape[0] - len(ind), 100, 100))
numlabelsToUse = np.zeros((imgs.shape[0] - len(ind) ))
index = 0
for i in range(imgs.shape[0]):
	good = 1
	for j in range(len(ind)):
		if i == ind[j]:
			good = 0
	
	if good == 1:
		imgsToUse[index] = imgs[i]
		numlabelsToUse[index] = numlabels[i]
		index += 1

imgs = imgsToUse
numlabels = numlabelsToUse.astype(int)

imgsFlat = imgs.reshape(( imgs.shape[0], imgs.shape[1]**2 ))
#imgs = resizeDat(datStr, dim)

pca = PCA(n_components=180)
imgPCA = pca.fit_transform(imgsFlat)
'''
invImg = pca.inverse_transform(imgPCA)

imgPlot = np.zeros((imgs.shape[0],imgs.shape[1], imgs.shape[2])) 
for i in range(10):
	imgPlot[2*i] = imgs[i]
	imgPlot[2*i+1] = invImg[i].reshape((100, 100))


dataPrep.PlotImgs(imgPlot, 10, 2)
'''

def visualize_scatter(data_2d, label_ids, labelnames, figsize=(20,20)):
	import matplotlib.pyplot as plt
	plt.figure(figsize=figsize)
	plt.grid()

	nb_classes = len(np.unique(label_ids))
	print 'number of classes', nb_classes

	for label_id in np.unique(label_ids):
		plt.scatter(data_2d[np.where(label_ids == label_id), 0],
		                data_2d[np.where(label_ids == label_id), 1],
		                marker='o',
		                color= plt.cm.Set1(label_id / float(nb_classes)),
		                linewidth='1',
		                alpha=0.8,
		                label=labelnames[label_id])
	plt.legend(loc='best')
	plt.show()


def visualize_scatter_with_images(X_2d_data, images, figsize=(45,45), image_zoom=1):
	from matplotlib.offsetbox import OffsetImage, AnnotationBbox
	import matplotlib.pyplot as plt
	from skimage.transform import resize

	fig, ax = plt.subplots(figsize=figsize)
	artists = []
	for xy, i in zip(X_2d_data, images):
		x0, y0 = xy
		i = resize(i, (25, 25))
		img = OffsetImage(i, zoom=image_zoom)
		ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
		artists.append(ax.add_artist(ab))
	ax.update_datalim(X_2d_data)
	ax.autoscale()
	plt.show()

def plotLatDim(encoder, dat, label):
	z_mean, _, _ = encoder.predict(dat, batch_size=g.bsize)
	plt.figure(figsize=(12, 10))
    	plt.scatter(z_mean[:, 0], z_mean[:, 1], c=label, s=4)
    	plt.colorbar()
    	plt.xlabel("z[0]")
    	plt.ylabel("z[1]")
    	plt.savefig('results/VAEhlCnnTr.png')
    	plt.show()


label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
	'Fat Hen', 'Loose Silky-bent','Maize', 'Scentless Mayweed', 'Shepherds Purse',
	'Small-flowered Cranesbill', 'Sugar beet']



'''
orderedY = []
for i in range(numlabels.shape[0]):
	orderedY.append(label[numlabels[i]])

print orderedY[0:5]
'''

if os.path.exists('data/tSNEresScaled.npy'):
	print 'Pulling saved run'
	tsne_result_scaled = np.load('data/tSNEresScaled.npy')
	print tsne_result_scaled.shape
else:
	print 'Running tsne'
	tsne = TSNE(n_components=2, perplexity=40.0)
	tsne_result = tsne.fit_transform(imgPCA)
	tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
	np.save('data/tSNEresScaled', tsne_result_scaled)


visualize_scatter(tsne_result_scaled, numlabels, label )

visualize_scatter_with_images(tsne_result_scaled, imgs, image_zoom=0.7)





