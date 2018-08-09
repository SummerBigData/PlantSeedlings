
import numpy as np
import dataPrep
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

# Definitions ---------- Definitions ---------- Definitions ---------- Definitions ---------- Definitions

def DatAug(xSt, unlSt, xtsne, unltsne):
	'''
	from sklearn.decomposition import PCA

	# Whiten the stamp images
	xcomb = np.concatenate((xSt, unlSt), axis=0)
	xcombFlat = xcomb.reshape((xcomb.shape[0], xcomb.shape[1]**2))

	pca = PCA(n_components=100)
	xCombPCA = pca.fit_transform(xcombFlat)
	
	xSt, unlSt = xCombPCA[ :xSt.shape[0] ], xCombPCA[ xSt.shape[0]: ]

	# Add the tSNE coordinates
	xSt = np.concatenate((xSt, xtsne), axis = 1)
	unlSt = np.concatenate((unlSt, unltsne), axis = 1)
	'''
	xSt = xtsne
	unlSt = unltsne
	return xSt, unlSt

# Constants ---------- Constants ---------- Constants ---------- Constants ---------- Constants

seed = 7
#np.random.seed(seed)

dim = 51 
featSize = 2
folds = 10
split = 0.2
cutoff= 0.90
epo = 500
bsize = 25

modelSpecs = 'cnnAugJustTSNEKERAS' #'cnnConv-16-16-32-32-Dense-64-32-12'
if not os.path.exists('weights/' + modelSpecs):
	os.makedirs('weights/' + modelSpecs)
	os.makedirs('predicts/' + modelSpecs)

label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
	'Fat Hen', 'Loose Silky-bent','Maize', 'Scentless Mayweed', 'Shepherds Purse',
	'Small-flowered Cranesbill', 'Sugar beet']

saveStr = modelSpecs+'/imgSize'+str(dim)+'Bsize'+str(bsize)+'Split'+str(split)+'k'
saveStr = 'weights/' + saveStr #+ '.hdf5' #'{epoch:02d}-{val_loss:.2f}.hdf5'

# Data Prep ---------- Data Prep ---------- Data Prep ---------- Data Prep ---------- Data Prep

# Get the images and labels
imgs, labels = dataPrep.getTrainDat(dim)
unlab = dataPrep.getTestDat(dim)

# Get the stamp images
imgsSt = dataPrep.resizeDat('data/trainImgsRes400stampBW'+str(200)+'Fixed.npy', dim)
unlabSt = dataPrep.resizeDat('data/testImgsRes400stampBW'+str(200)+'Fixed.npy', dim)

# Get the tSNE coordinates
tSNEdat = np.load('data/tSNEresScaledAllResults.npy') # 5544 x 2
tSNEimgs = tSNEdat[ :imgs.shape[0] ]
tSNEunlab = tSNEdat[ imgs.shape[0]: ]

# Augment the Data
xst, unlst = DatAug(imgsSt, unlabSt, tSNEimgs, tSNEunlab)
print 'Augmentation data size', xst.shape, unlst.shape

#x, y = dataPrep.shuffleData(imgs, labels, seed)
# Get the list of data for kfold testing
xtrSp, xsttrSp, ytrSp, xteSp, xstteSp, yteSp = dataPrep.DatSplitStamp(imgs, xst, labels, split, folds)


# KERAS NEURAL NETWORK

# Load or make Model
modelStr = 'models/'+ modelSpecs + str(dim)
'''
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
	model = dataPrep.getcnnKERAS(dim)
	model.save(modelStr)
	print 'Created and saved model.'
	print ' '
'''
model = dataPrep.getcnnKERAS(dim, featSize)
# Initialize the vectors to hold the final performance. tr loss, tr percent, te loss, te percent
results = np.zeros((folds, 4))
# Also initialize the predictions matrix
preds = np.zeros((folds, unlab.shape[0], 12))


# Do the following for each kfold
for i in range(folds):
	np.random.seed(seed+i)
	# Get the appropriate data for the ith kfold from the data split
	xtr, xsttr, ytr = xtrSp[i], xsttrSp[i], ytrSp[i]
	xte, xstte, yte = xteSp[i], xstteSp[i], yteSp[i]
	xtr = xtr.reshape((xtr.shape[0], dim, dim, 3))
	xte = xte.reshape((xte.shape[0], dim, dim, 3))
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
		scores = model.evaluate([xte, xstte], yte_binary, verbose=0)
		if scores[1] < cutoff:
			#os.remove(ithSaveStr)		# FIX
			print ' '
			print "Bad saved trial. Testing acc <"+str(cutoff)+"%. Rerunning ..."
			print ' '
			callbacks = dataPrep.get_callbacks(filepath=ithSaveStr, patience=60)
			# Fit the model
			model.fit([xtr, xsttr], ytr_binary,
				batch_size=bsize,
				epochs=epo,
				verbose=2,
				validation_data=([xte, xstte], yte_binary),
				callbacks=callbacks)

	else:
		print 'No saved trial for kfold.'
		callbacks = dataPrep.get_callbacks(filepath=ithSaveStr, patience=60)
		# Fit the model
		model.fit([xtr, xsttr], ytr_binary,
			batch_size=bsize,
			epochs=epo,
			verbose=2,
			validation_data=([xte, xstte], yte_binary),
			callbacks=callbacks)


	# Evaluate the model.Calculate the scores on the training and testing data
	model.load_weights(ithSaveStr)
	# Training
	scores = model.evaluate([xtr, xsttr], ytr_binary, verbose=0)
	results[i, 0], results[i, 1] = scores[0], scores[1]
	# Testing
	scores = model.evaluate([xte, xstte], yte_binary, verbose=0)
	results[i, 2], results[i, 3] = scores[0], scores[1]
	# Unlabeled predictions
	preds[i] = model.predict([unlab, unlst])

	print 'Training loss for fold',i,'is', results[i,0],'with percent',results[i,1]*100
	print 'Testing loss for fold',i,'is', results[i,2],'with percent',results[i,3]*100
	print ' '

print 'Prediction dims', preds.shape
np.save('predicts/'+modelSpecs+'/predDim'+str(dim)+'Folds'+str(folds)+'split'+str(split)+'8-9', preds)

print 'Done training kfolds. Results:'
print results






