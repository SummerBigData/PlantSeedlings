import numpy as np
import pandas as pd
import os

import dataPrep

#modelSpecs = 'cnnConv-16-16-32-32-Dense-64-32-12'
modelSpecs = 'cnnConv-16-32-64-Dense-64-32-12-CB-KERAS-dim-65-date-8-17onlyTSNE'
predDat = '/'+'predDim65Folds10split0.28-13.npy'
submitName = 'Folds10Split0.2Dim65tSNEDate8-17.csv'






label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
	'Fat Hen', 'Loose Silky-bent','Maize', 'Scentless Mayweed', 'Shepherds Purse',
	'Small-flowered Cranesbill', 'Sugar beet']


def prepSub(pred):
	predMax = np.argmax(pred, axis=1)

	labeledPred = []
	for i in range(predMax.shape[0]):
		labeledPred.append( label[predMax[i]] )
	return labeledPred

def genSub(guess, names, stat):
	guess = prepSub(guess)

	submission = pd.DataFrame()
	submission['file']= names
	submission['species']= guess
	submission.to_csv('submits/sub' + stat + submitName, index=False)




# Predictions are (#folds, 794, 12)
preds = np.load('predicts/'+ modelSpecs + predDat)
# Names are (794,)
names = np.load('data/testNames.npy')




meanPred = np.mean(preds, axis = 0)
#medPred = np.median(preds, axis = 0)

genSub(meanPred, names, 'Mean')
#genSub(medPred, names, 'Median')




