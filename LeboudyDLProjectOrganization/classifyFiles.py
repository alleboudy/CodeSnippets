#the testing and classification script
import numpy as np
import utilities
import settings
from leboudynet import create_model
import pickle
import os

inputRoot=#Root directory where folders of possible raw data to classify exists
aggrigatedScoreResults=#folder 2 store the results
testweights=settings.testweights
classesNames = settings.classesNames#dictionary with each class assigned an integer
topClassesPerFile = settings.topClassesPerFile#top classes considered per file
reverseclassesNames = {v: k for k, v in classesNames.items()}#dictionary with integers to corresponding class names
labels = [] 
files = []
model =create_model(testweights)

def LoadAndPreProcessFile(filePath):
	with open(filePath, encoding="utf8") as f:#could be an imread with open cv as well
						filenpy=[] #should hold file contents eventually
	return np.asarray(filenpy)





for foldername in os.listdir(inputRoot):#looping over a 2 levels hierarchy of folders of testing data
			folder = os.path.join(inputRoot, foldername)
			if os.path.isdir(folder):
				outFolder = os.path.join(mirrorRoot, foldername)
				if not os.path.exists(outFolder):
					os.makedirs(outFolder)
				outFolderDict = dict()#holds per directory its files' classifications
				for filename in os.listdir(folder):
					file = os.path.join(folder, filename)
					if not os.path.isdir(file):
						out = model.predict(np.expand_dims(LoadAndPreProcessFile(file), axis=0))
						outfilePath = os.path.join(outFolder, filename)
						fileDict = dict()
						for clss in out[0].argsort()[-topClassesPerFile:][::-1]:#getting top classes per file
							fileDict[reverseclassesNames[clss]] = out[0][clss]#storing their scores
						outFolderDict[filename]=fileDict
						pickle.dump( fileDict, open( outfilePath.replace(img or txt or whatever input file extension,'.p'), "wb" ) )

