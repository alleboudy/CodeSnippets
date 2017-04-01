#shall include our data batches generator
import numpy as np
import random
import settings
import os
dataset = settings.traindata  # 'dataset_train.txt'
batchSize = settings.batchSize
classesNames = settings.classesNames#dictionay of class,integerIndex
maxNumberOfExamplesPerClass = settings.maxNumberOfExamplesPerClass
classesNamesVsCount = dict()

def get_data():
	print ("getting data")
	files_batch = []
	labels = []
	for file in os.listdir(dataset):
			if file.endswith(XXX):
				#read it or do whatever
				if "condition on file content and length":
					hotEncodings = np.zeros(len(classesNames))
					for clss in file.split('_')[0].split('.'):#file name should be clss1.clss2.clss3_fileid.txt
						if clss not in classesNamesVsCount:
							classesNamesVsCount[clss]=0
						elif classesNamesVsCount[clss]>maxNumberOfExamplesPerClass:
							continue
						classesNamesVsCount[clss]+=1

						hotEncodings[classesNames[clss]] = 1
					if not hotEncodings.any(): #if it is all zeroes, skip it
						continue
					files_batch.append()#file content
					labels.append(hotEncodings)
	settings.totalNumberOfUniqueExamples  = str(len(files_batch))
	print('totalnumber of examples: ' +settings.totalNumberOfUniqueExamples  )
	return (files_batch, labels)


def gen_data(source):
	print ("in single generator")
	while True:
		indices = list(range(len(source[0])))
		random.shuffle(indices)
		for i in indices:
			fileMatrix = source[0][i]
			label = source[1][i]
			yield fileMatrix, label

def get_data_examples(source):
	print ("in data example generator")
	indices = list(range(len(source[0])))
	#random.shuffle(indices)
	for i in indices:
		fileMatrix = source[0][i]
		label = source[1][i]
		yield fileMatrix, label


def gen_data_batch(source):
	print ("in batch generator")
	data_gen = gen_data(source)
	while True:
		files_batch = []
		labels_batch = []
		for _ in range(batchSize):
			fileMatrix, label = next(data_gen)
			files_batch.append(fileMatrix)
			labels_batch.append(label)
		print('totalnumber of examples: ' +settings.totalNumberOfUniqueExamples  )
		yield np.array(files_batch),np.asarray(labels_batch)