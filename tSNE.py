import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import dataPrep


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




label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
	'Fat Hen', 'Loose Silky-bent','Maize', 'Scentless Mayweed', 'Shepherds Purse',
	'Small-flowered Cranesbill', 'Sugar beet']



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





