'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
			dataSet: size m data set of known vectors (NxM)
			labels: data set labels (1xM vector)
			k: number of neighbors to use for comparison (should be an odd number)
			
Output:     the most popular class label

@author: pbharrin



For every point in our dataset:
	Calculate the distance between inX and the current point
	Sort the distances in increasing order
	Take k items with lowest distance to inX
	Find the majority class among these items
	Return the majority class as our prediction for the class of inX


>python
>> import kNN
>> group,labels = kNN.createDataSet()
>> kNN.classify0([0,0],group,labels,3)    (result 'B')
'''


from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX, dataSet, labels, k):
	# Distance calculation
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort() 
	# Voting with lowest k distances
	classCount={}          
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	# Sort dictionary
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]




