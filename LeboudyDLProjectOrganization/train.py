#the training script
import utilities
import numpy as np
from leboudynet import create_model
import keras
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K
import time
import settings
outputWeightspath =settings.outputWeightspath
startweight= settings.startweight
nb_epochs = settings.nb_epochs
batchSize=utilities.batchSize
historyloglocation  = './{}_traininghistory.txt'.format(str(time.time()))
trainedmodelPath=settings.trainedmodelPath
intermediateModePath=settings.intermediateModePath

print ("creating the model")
model =create_model(startweight)


datasource = utilities.get_data()

data_gen = utilities.gen_data_batch(datasource)

for i in range(nb_epochs):

	X_batch, Y_batch = next(data_gen)	
	history = model.fit(X_batch,Y_batch,
          nb_epoch=1,batch_size=batchSize)
	print ('epoch: ', i)
	print ('loss: ',history.history['loss'][0])
	with open(historyloglocation,"a+") as f:
		f.write('{},{}\n'.format(str(i), str(history.history['loss'][0])))

	if i%25==0:
		print ('saved trained weights in: ')
		print (outputWeightspath)
		model.save_weights(outputWeightspath,overwrite=True)
		model.save(intermediateModePath)


model.save_weights(outputWeightspath)
model.save(trainedmodelPath)