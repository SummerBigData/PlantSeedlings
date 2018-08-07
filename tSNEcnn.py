
import numpy as np
import dataPrep
import os
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

'''
dim = 200
# Get the statistics of an image or many images
def GetStats(imgs, oneImg):	# oneImg is 1 if true, 0 if false
	imgsM = np.zeros((3))
	imgsS = np.zeros((3))
	if oneImg == 1:
		for i in range(3):
			imgsM[i] = np.mean(imgs[:,:,i])
			imgsS[i] = np.std(imgs[:,:,i])
	else:
		for i in range(3):
			imgsM[i] = np.mean(imgs[:,:,:,i])
			imgsS[i] = np.std(imgs[:,:,:,i])	
	return imgsM, imgsS

def FixColor(img, means, stds):
	wrongM, wrongS = GetStats(img, 1)
	fixedImg = np.zeros((img.shape))
	for i in range(3):
		temp = (img[:,:,i] - wrongM[i]) / wrongS[i]
		fixedImg[:,:,i] = temp * stds[i] + means[i]
	return fixedImg




#imgs, labels = dataPrep.getTrainDat(dim)
unlab = dataPrep.getTestDat(dim)
imgs = unlab
#imgs, labels = dataPrep.shuffleData(imgs, labels, seed)

imgsMean, imgsStd = GetStats(imgs, 0)

imgMask = np.zeros((imgs.shape[0], dim, dim))
imgZero = []
wrongImgs = []
for i in range(imgs.shape[0]):	# FIX
	imgMask[i] = dataPrep.getPlantMask(imgs[i], 36)
	if imgMask[i].max() < 0.001:
		print 'Fixed here:' , i
		wrongImgs.append(i)
		imgs[i] = FixColor(imgs[i], imgsMean, imgsStd)
		imgMask[i] = dataPrep.getPlantMask(imgs[i], 36)

stampImgs = np.zeros(imgMask.shape)

for i in range(imgs.shape[0]):
	avgImg = (imgs[i,:,:,0] + imgs[i,:,:,1] + imgs[i,:,:,2]) / 3.0
	stampImgs[i] = np.multiply(avgImg, imgMask[i])


np.save('data/testImgsRes400stampBW'+str(dim)+'Fixed.npy', stampImgs)
print 'Done making stamp'
'''


seed = 7
dim = 200
folds = 10
split = 0.2
cutoff= 0.90
epo = 500
bsize = 25
modelSpecs = 'tsneCnnConv-16-16-32-32-Dense-64-32-12'



# Get the usual images
imgsRGB, labels = dataPrep.getTrainDat(dim)
unlabRGB = dataPrep.getTestDat(dim)

# Get the stamp images and shufle them
imgs = dataPrep.resizeDat('data/trainImgsRes400stampBW'+str(200)+'Fixed.npy', dim)
x, y = dataPrep.shuffleData(imgs, labels, seed)

unlab = dataPrep.resizeDat('data/testImgsRes400stampBW'+str(200)+'Fixed.npy', dim)

# Get the list of data for kfold testing
xtrSp, ytrSp, xteSp, yteSp = dataPrep.DatSplit(x, y, split, folds)

# Labels in case they are needed
label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
	'Fat Hen', 'Loose Silky-bent','Maize', 'Scentless Mayweed', 'Shepherds Purse',
	'Small-flowered Cranesbill', 'Sugar beet']


# Pick a seed
np.random.seed(seed)

# Create the save string used for the weights
saveStr = modelSpecs+'/imgSize'+str(dim)+'Bsize'+str(bsize)+'k'
saveStr = 'weights/' + saveStr #+ '.hdf5' #'{epoch:02d}-{val_loss:.2f}.hdf5'


# KERAS NEURAL NETWORK

# Load or make Model
modelStr = 'models/'+ modelSpecs + str(dim)
#os.remove(modelStr)	# FIX
if os.path.exists(modelStr):
	print ' '
	print 'Found model'
	print ' '
	from keras.models import load_model
	model = load_model(modelStr)
else:
	print ' '
	print 'No saved model. Generating...'
	model = dataPrep.getTsneCnn(dim)
	model.save(modelStr)
	print 'Created and saved model.'
	print ' '

# Initialize the vectors to hold the final performance. tr loss, tr percent, te loss, te percent
results = np.zeros((folds, 4))
# Also initialize the predictions matrix
preds = np.zeros((folds, unlab.shape[0], 12))


# Do the following for each kfold
for i in range(folds):
	np.random.seed(seed+i)
	# Get the appropriate data for the ith kfold from the data split
	xtr, ytr = xtrSp[i], ytrSp[i]
	xte, yte = xteSp[i], yteSp[i]
	xtr = xtr.reshape((xtr.shape[0], dim, dim, 1))
	xte = xte.reshape((xte.shape[0], dim, dim, 1))
	# Need labels in matrix shape for the model, so convert to binary
	ytr_binary = to_categorical(ytr)
	yte_binary = to_categorical(yte)

	# The exact save string can be made, now we know the kfold
	ithSaveStr = saveStr+str(i)+'.hdf5'
	# Pull good weights or run the cnn
	if os.path.exists(ithSaveStr):
		print ' '
		print 'Pulling kfold', i, 'from previous runs'
		model.load_weights(ithSaveStr)
		scores = model.evaluate(xte, yte_binary, verbose=0)
		if scores[1] < cutoff:
			os.remove(ithSaveStr)
			print ' '
			print "Bad saved trial. Testing acc <"+str(cutoff)+"%. Rerunning ..."
			print ' '
			callbacks = dataPrep.get_callbacks(filepath=ithSaveStr, patience=60)
			# Fit the model
			model.fit(xtr, ytr_binary,
				batch_size=bsize,
				epochs=epo,
				verbose=2,
				validation_data=(xte, yte_binary),
				callbacks=callbacks)

	else:
		print 'No saved trial for kfold.'
		callbacks = dataPrep.get_callbacks(filepath=ithSaveStr, patience=60)
		# Fit the model
		model.fit(xtr, ytr_binary,
			batch_size=bsize,
			epochs=epo,
			verbose=2,
			validation_data=(xte, yte_binary),
			callbacks=callbacks)


	# Evaluate the model.Calculate the scores on the training and testing data
	model.load_weights(ithSaveStr)
	# Training
	scores = model.evaluate(xtr, ytr_binary, verbose=0)
	results[i, 0], results[i, 1] = scores[0], scores[1]
	# Testing
	scores = model.evaluate(xte, yte_binary, verbose=0)
	results[i, 2], results[i, 3] = scores[0], scores[1]
	# Unlabeled predictions
	preds[i] = model.predict(unlab)

	print 'Training loss for fold',i,'is', results[i,0],'with percent',results[i,1]*100
	print 'Testing loss for fold',i,'is', results[i,2],'with percent',results[i,3]*100
	print ' '

print 'Prediction dims', preds.shape
np.save('predicts/'+modelSpecs+'/predDim'+str(dim)+'Folds'+str(folds)+'split'+str(split)+'8-7', preds)

print 'Done training kfolds. Results:'
print results







