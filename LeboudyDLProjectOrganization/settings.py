#global variables and hyperparameters live here
import time
import pickle
import os
import keras
import tensorflow as tf
#from keras.optimizers import SGD
optimizer =keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(learning_rate=0.5, beta1=0.9, beta2=0.999, epsilon=0.1, use_locking=False, name='Adam'))
#SGD(lr=0.01, momentum=0.80, decay=1e-6, nesterov=True)
#keras.optimizers.TFOptimizer(tf.train.AdagradOptimizer(learning_rate=1))

batchSize = 70
nb_epochs=30000
validationSplit=0.1
traindata=''

#----- Loading classes names
loadClassesNames = True
classNamesDictLocation='classNamesDict.p'
def getClassNames():
	if loadClassesNames:
		clsNamesDict = pickle.load( open( classNamesDictLocation, "rb" ) )
		return clsNamesDict
		#otherwise preprocess the available names and load them


classesNames = getClassNames()




#------
historyloglocation =  'lossLogs/{}.txt'.format(time.strftime("%d-%m-%Y_%H-%M-%S"))
testdata = ''
post='descriptiveNameForTheParameters'
outputWeightspath ='intermediateModels/_'+post+'.h5' #'intermediateModels/{epoch:02d}-{val_loss:.2f}_'+post+'.h5'
testweights = ''

