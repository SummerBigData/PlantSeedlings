import numpy as np
import os

'''
4750 images total
largest x = 3457
smallest x = 49

largest y = 3991
smallest y = 49
'''


def GenTrainDat():
	from PIL import Image
	label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
		'Fat Hen', 'Loose Silky-bent','Maize', 'Scentless Mayweed', 'Shepherds Purse',
		'Small-flowered Cranesbill', 'Sugar beet']

	imgs = []
	#numbEach = np.array((12))
	numLabel = []
	for i in range(len(label)):
		folder = 'data/train/' + label[i]
		for filename in os.listdir(folder):
			img = Image.open(folder+'/'+filename)
			img = np.array(img)

			imgs.append(img)
			numLabel.append(i)
	return imgs, numLabel

'''
def PlotImgs(imgs):
	fig, axes = plt.subplots(nrows=12, ncols=8)
	ax = axes.ravel()
	for i in range(imgs.shape[0]):
		ax[i].imshow(imgs[i])
		#ax[i].set_title(label[labels[i]])
	#plt.tight_layout()
	plt.show()

#PlotImgs(imgs, labels, 36)
numSpecies = 8
locations = np.zeros((12, numSpecies)).astype(int)

for i in range(12):
	locations[i] = np.where(labels == i)[0][0:numSpecies]

imgPlot = np.zeros((12*numSpecies, imgs.shape[1], imgs.shape[1], 3))

for i in range(12):
	for j in range(numSpecies):
		imgPlot[i*numSpecies+j] = imgs[locations[i, j]]

PlotImgs(imgPlot)

print locations
'''

def resizeTrainDat(dim):
	from skimage.transform import resize

	imgs = np.load('data/trainImgs'+'.npy')
	imgResized = np.zeros(( len(imgs), dim, dim, 3))
	
	for i in range(len(imgs)):
		imgResized[i] = resize(imgs[i], (dim, dim, 3))#, anti_aliasing=True)

	return imgResized

'''
imgResized = resizeTrainDat(100)
np.save('data/trainImgsResized100', imgResized)
'''

def getTrainDat(dim):
	saveStr = 'data/trainImgsResized'+str(dim)+'.npy'
	if os.path.exists(saveStr):
		print 'Found data with correct size'
		imgs = np.load('data/trainImgsResized'+str(dim)+'.npy')
		labels = np.load('data/trainLabels'+'.npy')
	else:
		print 'Did not find data with correct size. Generating...'
		print ' '
		imgResized = resizeTrainDat(dim)
		np.save('data/trainImgsResized'+str(dim), imgResized)
		print 'Done. Loading images'
		imgs = np.load('data/trainImgsResized'+str(dim)+'.npy')
		labels = np.load('data/trainLabels'+'.npy')

	return imgs, labels.astype(int)


def shuffleData(xtr, ytr, ind):
	np.random.seed(ind)
	imgsize = xtr.shape[1]
	# Linearize the x features and columnize the y data
	xtr = xtr.reshape((xtr.shape[0], imgsize*imgsize*3))
	ytr = ytr.reshape((xtr.shape[0], 1))
	# Stick them together and shuffle them
	augdat = np.hstack((ytr, xtr))	#(2000, 16876)
	np.random.shuffle(augdat)
	# Seperate augdat and reshape back to original dimensions
	xtrshuffle = augdat[:, 1:].reshape((xtr.shape[0], imgsize, imgsize, 3))
	ytrshuffle = np.ravel(augdat[:, 0])
	return xtrshuffle, ytrshuffle.astype(int)


def getcnn(imgsize):
	from keras.models import Model
	from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Input
	from keras.optimizers import Adam	
	from keras.layers.normalization import BatchNormalization
	#from keras import regularizers

	p_activation = "elu"
	input_1 = Input(shape=(imgsize, imgsize, 3))
	#input_2 = Input(shape=[1], name="angle")
	c1 = Conv2D(16,kernel_size = (3,3),activation=p_activation)(input_1)

	c2 = Conv2D(16, kernel_size = (3,3), activation=p_activation) (c1)
	c2 = MaxPooling2D((2,2)) (c2)
	c2 = Dropout(0.2)(c2)

	c3 = Conv2D(32, kernel_size = (3,3), activation=p_activation) (c2)

	c4 = Conv2D(32, kernel_size = (3,3), activation=p_activation) (c3)
	c4 = MaxPooling2D((2,2)) (c4)
	c4 = Dropout(0.2)(c4)

	c5 = Conv2D(64, kernel_size = (3,3), activation=p_activation) (c4)

	c6 = Conv2D(64, kernel_size = (3,3), activation=p_activation) (c5)
	c6 = MaxPooling2D((2,2)) (c6)
	c6 = Dropout(0.2)(c6)

	c7 = Conv2D(128, kernel_size = (3,3), activation=p_activation) (c6)
	c7 = MaxPooling2D((2,2)) (c7)
	c7 = Dropout(0.2)(c7)
	c8 = c7
	
	'''
	c8 = Conv2D(128, kernel_size = (3,3), activation=p_activation)(c7)
	c8 = MaxPooling2D((2,2)) (c8)
	c8 = Dropout(0.2)(c8)
	#c8 = GlobalMaxPooling2D() (c8)
	'''
	#img_concat =(Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))
	d = Flatten()(c8)
	d = BatchNormalization()(d)
	d = Dense(256, activation=p_activation)(d)
	d = Dropout(0.2)(d)
	d = Dense(64, activation=p_activation)(d) 
	d = Dropout(0.2)(d)
	output = Dense(12, activation="sigmoid")(d)

	model = Model(input_1,  output)
	optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
	model.summary()
	return model


# We choose a high patience so the algorthim keeps searching even after finding a maximum
def get_callbacks(filepath, patience=8):
	from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
	
	es = EarlyStopping('val_loss', patience=patience, mode="min")
	msave = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True,save_weights_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=7,min_lr=0.0005,mode="min")
	return [es, msave, reduce_lr]


def DatSplit(x, y, numb): # numb example: 0.75
	numb *= x.shape[0]
	numb = int(numb)
	xtr = x[0:numb]
	xte = x[numb:]
	ytr = y[0:numb]
	yte = y[numb:]

	return xtr, ytr, xte, yte

def Norm(mat, nMin, nMax):
	# Calculate the old min, max and convert to float values
	Min = np.amin(mat).astype(float)
	Max = np.amax(mat).astype(float)
	nMin = nMin+0.0
	nMax = nMax+0.0
	# Calculate the new normalized matrix
	normMat = ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin
	return normMat, Min, Max





imgs, labels = getTrainDat(100)


def PlotImgs(imgs, numImgs):
	fig, axes = plt.subplots(nrows=1, ncols=numImgs)
	ax = axes.ravel()
	for i in range(imgs.shape[0]):
		ax[i].imshow(imgs[i])
		#ax[i].set_title(label[labels[i]])
	#plt.tight_layout()
	plt.show()

imgsDC = np.zeros((imgs.shape))
imgsHess = np.zeros((4, 100, 100, 3))

from skimage.filters import hessian



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




import matplotlib.pyplot as plt
imgPlot = np.zeros((5, 100, 100, 3))
imgPlot[0, :, :, 0] = imgs[0, :, :, 0]
imgPlot[1, :, :, 1] = imgs[0, :, :, 1]
imgPlot[2, :, :, 2] = imgs[0, :, :, 2]
imgPlot[3, :, :, :] = imgs[0]
imgPlot[4, :, :, :] = imgsHess[0]

PlotImgs(imgPlot, 5)







