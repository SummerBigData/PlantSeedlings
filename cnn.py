
import numpy as np
import dataPrep
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

seed = 7
dim = 100
folds = 3
split = 0.2


imgs, labels = dataPrep.getTrainDat(dim)
x, y = dataPrep.shuffleData(imgs, labels, seed)
xtrSp, ytrSp, xteSp, yteSp = dataPrep.DatSplit(x, y, split, folds)

label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
	'Fat Hen', 'Loose Silky-bent','Maize', 'Scentless Mayweed', 'Shepherds Purse',
	'Small-flowered Cranesbill', 'Sugar beet']



np.random.seed(seed)

epo = 300
bsize = 25
saveStr = 'plainCnn/imgSize'+str(dim)+'Bsize'+str(bsize)
saveStr = 'weights/' + saveStr + '.hdf5' #'{epoch:02d}-{val_loss:.2f}.hdf5'


# KERAS NEURAL NETWORK

model = dataPrep.getcnn(dim)
#os.remove('models/cnnModel' + str(dim) )
model.save('models/cnnModel' + str(dim) )

callbacks = dataPrep.get_callbacks(filepath=saveStr, patience=40)

# Need labels in matrix shape for the model, so convert to binary
xtr, ytr = xtrSp[0], ytrSp[0]
xte, yte = xteSp[0], yteSp[0]

ytr_binary = to_categorical(ytr)
yte_binary = to_categorical(yte)
print ytr_binary.shape, yte_binary.shape

# Fit the model
model.fit(xtr, ytr_binary,
	batch_size=bsize,
	epochs=epo,
	verbose=2,
	validation_data=(xte, yte_binary),
	callbacks=callbacks)

# evaluate the model
model.load_weights(saveStr)

# Calculate the scores on the training and testing data
results = np.zeros((2, 2))
# Training
scores = model.evaluate(xtr, ytr_binary, verbose=0)
results[0, 0] = scores[0]
results[0, 1] = scores[1]
# Testing
scores = model.evaluate(xte, yte_binary, verbose=0)
results[1, 0] = scores[0]
results[1, 1] = scores[1]

#prediction = model.predict(unlab)
	
print 'Training percent for iter', 0, 'is', scores[0,1]*100, 'with log loss', scores[0,0]
print 'Testing percent for iter', 0, 'is', scores[1,1]*100, 'with log loss', scores[1,0]
print ' '



