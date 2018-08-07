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





def PlotImgs(imgs, nrows, ncols):
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
	ax = axes.ravel()
	for i in range(nrows*ncols):
		ax[i].imshow(imgs[i])
		ax[i].axes.get_yaxis().set_visible(False)
		ax[i].axes.get_xaxis().set_visible(False)
		#ax[i].set_title(label[labels[i]])
	#plt.tight_layout()
	plt.show()

def resizeDat(datStr, dim):
	from skimage.transform import resize

	imgs = np.load(datStr) #'data/trainImgs'+'.npy'
	dimensions = len(imgs.shape)
	if dimensions == 3:
		imgResized = np.zeros((len(imgs), dim, dim))
		for i in range(len(imgs)):
			imgResized[i] = resize(imgs[i], (dim, dim))#, anti_aliasing=True)
		return imgResized
	else:
		imgResized = np.zeros(( len(imgs), dim, dim, 3))
		for i in range(len(imgs)):
			imgResized[i] = resize(imgs[i], (dim, dim, 3))#, anti_aliasing=True)
		return imgResized
	return

def GenTestDat():
	from PIL import Image

	imgs =[]
	names = []
	folder = 'data/test'
	for filename in os.listdir(folder):
		img = Image.open(folder+'/'+filename)
		img = np.array(img)
		imgs.append(img)
		names.append(filename)
	np.save('data/testImgs', imgs)
	np.save('data/testNames', names)
	imgRes = resizeDat('data/testImgs.npy', 400)
	np.save('data/testImgsResized400', imgRes)
	return


def getTrainDat(dim):
	saveStr = 'data/trainImgsResized'+str(dim)+'.npy'
	if os.path.exists(saveStr):
		print ' '
		print 'Found train data with correct size'
		print ' '
		imgs = np.load(saveStr)
		labels = np.load('data/trainLabels'+'.npy')
	else:
		print ' '
		print 'Did not find train data with correct size. Generating...'
		print ' '
		imgResized = resizeDat('data/trainImgs.npy',dim)
		np.save('data/trainImgsResized'+str(dim), imgResized)
		print 'Done. Loading train images'
		print ' '
		imgs = np.load(saveStr)
		labels = np.load('data/trainLabels'+'.npy')
	return imgs, labels.astype(int)

def getTestDat(dim):
	saveStr = 'data/testImgsResized'+str(dim)+'.npy'
	if os.path.exists(saveStr):
		print ' '
		print 'Found test data with correct size'
		print ' '
		imgs = np.load(saveStr)
	else:
		print ' '
		print 'Did not find test data with correct size. Generating...'
		print ' '
		imgResized = resizeDat('data/testImgs.npy',dim)
		np.save('data/testImgsResized'+str(dim), imgResized)
		print 'Done. Loading test images'
		print ' '
		imgs = np.load(saveStr)
	return imgs

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
	from keras.optimizers import Adam, SGD	
	from keras.layers.normalization import BatchNormalization
	#from keras import regularizers

	p_activation = "relu"
	input_1 = Input(shape=(imgsize, imgsize, 3))
	#input_2 = Input(shape=[1], name="angle")
	
	convFilters = [16, 16, 32, 32]
	
	c = input_1
	for i in range(len(convFilters)):
		c = Conv2D(convFilters[i], kernel_size = (3,3), activation=p_activation) (c)
		c = MaxPooling2D((2,2)) (c)
		c = Dropout(0.2)(c)

	d = Flatten()(c)
	d = BatchNormalization()(d)

	denseFilters = [64, 32]
	
	for i in range(len(denseFilters)):
		d = Dense(denseFilters[i], activation=p_activation)(d)
		d = Dropout(0.2)(d)
	
	output = Dense(12, activation="softmax")(d)

	model = Model(input_1,  output)

	#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	optimizer = Adam(lr=1e-4) #1e-4
	#optimizer = SGD(lr=1e-1, momentum=0.9, nesterov=True)
	model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
	model.summary()
	return model


# We choose a high patience so the algorthim keeps searching even after finding a maximum
def get_callbacks(filepath, patience=8):
	from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
	
	es = EarlyStopping('loss', patience=patience, mode="min")
	msave = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True,save_weights_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=7,min_lr=5e-7,mode="min")
	return [es, msave, reduce_lr]


def DatSplit(x, y, testSplit, folds): # split example: 0.25 for  3/1 train/test split
	'''
	numb *= x.shape[0]
	numb = int(numb)
	xtr = x[0:numb]
	xte = x[numb:]
	ytr = y[0:numb]
	yte = y[numb:]	
	'''
	from sklearn.cross_validation import StratifiedShuffleSplit
	sss = StratifiedShuffleSplit(y, folds, test_size=testSplit, random_state=0)
	
	xtrSplit = []
	xteSplit = []
	ytrSplit = []
	yteSplit = []
	for train_index, test_index in sss:
		xtrSplit.append( x[train_index] )
		xteSplit.append( x[test_index] )
	
		ytrSplit.append( y[train_index] )
		yteSplit.append( y[test_index] )


	return xtrSplit, ytrSplit, xteSplit, yteSplit

def Norm(mat, nMin, nMax):
	# Calculate the old min, max and convert to float values
	Min = np.amin(mat).astype(float)
	Max = np.amax(mat).astype(float)
	if np.abs(Max - Min) < 1e-8:
		print 'Max and Min less than 1e-8 apart'
		return
	else:
		nMin = nMin+0.0
		nMax = nMax+0.0
		# Calculate the new normalized matrix
		normMat = ((mat - Min) / (Max - Min)) * (nMax - nMin) + nMin
	return normMat, Min, Max

def getPlantMask(image, sensitivity): # 36
	import cv2
	img = (image*255.0).astype(np.uint8)
	image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#sensitivity = 35
	lower_hsv = np.array([60 - sensitivity, 100, 50])
	upper_hsv = np.array([60 + sensitivity, 255, 255])

	mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	def sharpen_image(image):
		image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
		image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
		return image_sharp

	sharpmask = sharpen_image(mask) / 255.0
	return sharpmask






