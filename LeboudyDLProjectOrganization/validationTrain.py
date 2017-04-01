#the training script with automatic validation
import utilities
import numpy as np
from leboudynet import create_model
import keras
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K
import time
import settings
from keras.callbacks import ModelCheckpoint
outputWeightspath =settings.outputWeightspath
startweight= settings.startweight
nb_epochs = settings.nb_epochs
batchSize=utilities.batchSize
historyloglocation =  settings.historyloglocation
validationSplit = settings.validationSplit



class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		with open(historyloglocation,"a+") as f:
			f.write('{},{}\n'.format('val_loss', 'loss'))

	def on_batch_end(self, batch, logs={}):
		#print (self.params)
		pass

	def on_epoch_end(self, epoch, logs={}):
		with open(historyloglocation,"a+") as f:
			f.write('{},{}\n'.format(logs.get('val_loss'), logs.get('loss')))



print ("creating the model")
model =create_model(startweight)


datasource = utilities.get_data()




allX = []
allY =[]
for X,Y in utilities.get_data_examples(datasource):
	allX.append(X)
	allY.append(Y)

checkpointer = ModelCheckpoint(filepath=outputWeightspath, verbose=1, save_best_only=True,monitor='val_loss')
history = LossHistory()
model.fit(np.asarray(allX),np.asarray(allY),
	  epochs=nb_epochs,batch_size=batchSize,validation_split=validationSplit,callbacks=[history,checkpointer])



