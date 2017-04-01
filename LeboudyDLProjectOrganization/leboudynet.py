#this is where our model lives
import numpy as np
from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, Merge, Reshape, Activation,BatchNormalization
from keras.models import Model,Sequential
from keras.regularizers import l2
import settings
optm = settings.optimizer




def create_model(weights_path=None):
	model = Sequential()
	model.compile(loss='', optimizer=optm)

	if weights_path:
		print("using weights")
		print(weights_path)
		model.load_weights(weights_path,by_name=True)
	
	return model


