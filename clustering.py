import pandas as pd
import numpy as np
from numpy import *
# import time
import matplotlib.pyplot as plt
# from pandas import DataFrame
from scipy.spatial.distance import pdist

# dna = pd.read_csv('C:/Users/fader/Desktop/clustering/codon_usage.csv')
dna = pd.read_csv('codon_usage.csv')

# print(dna['Kingdom'].value_counts())
# print(dna)


dna.drop(columns=['DNAtype','SpeciesID','Ncodons','SpeciesName'],inplace=True)
# print(dna)

data = dna.iloc[:,1:].astype(float)
# print(data)


# calculate Euclidean distance
def euclDistance(vector1, vector2):
	return sqrt(sum(power(vector2 - vector1, 2)))

# calculate Manhattan distance
def manhattanDistance(vector1, vector2):
	return sum(abs(vector1-vector2))

# calculate Cosine distance
def cosineDistance(vector1, vector2):
	return pdist(np.vstack([vector1, vector2]), 'cosine')[0]

# init centroids with random samples
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids

# k-means cluster
def kmeans(dataSet, k, dis_standard):
	numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
	clusterAssment = mat(zeros((numSamples, 2)))
	clusterChanged = True

	## step 1: init centroids
	centroids = initCentroids(dataSet, k)

	while clusterChanged:
		clusterChanged = False
		## for each sample
		for i in range(numSamples):
			minDist  = 100000.0
			minIndex = 0
			## for each centroid
			## step 2: find the centroid who is closest
			for j in range(k):
				if dis_standard == 0:
					distance = euclDistance(centroids[j, :], dataSet[i, :])
				elif dis_standard == 1:
					distance = manhattanDistance(centroids[j, :], dataSet[i, :])
				elif dis_standard ==2:
					distance = cosineDistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist  = distance
					minIndex = j
			
			## step 3: update its cluster
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist**2

		## step 4: update centroids
		for j in range(k):
			pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] = mean(pointsInCluster, axis = 0)

	# print('Congratulations, cluster complete!')
	return centroids, clusterAssment


# k-medians cluster
def kmedians(dataSet, k, dis_standard):
	numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
	clusterAssment = mat(zeros((numSamples, 2)))
	clusterChanged = True

	## step 1: init centroids
	centroids = initCentroids(dataSet, k)

	while clusterChanged:
		clusterChanged = False
		## for each sample
		for i in range(numSamples):
			minDist  = 100000.0
			minIndex = 0
			## for each centroid
			## step 2: find the centroid who is closest
			for j in range(k):
				if dis_standard == 0:
					distance = euclDistance(centroids[j, :], dataSet[i, :])
				elif dis_standard == 1:
					distance = manhattanDistance(centroids[j, :], dataSet[i, :])
				elif dis_standard ==2:
					distance = cosineDistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist  = distance
					minIndex = j
			
			## step 3: update its cluster
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist**2

		## step 4: update centroids
		for j in range(k):
			pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] = median(pointsInCluster, axis = 0)

	# print('Congratulations, cluster complete!')
	return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
	numSamples, dim = dataSet.shape
	if dim != 2:
		print("Sorry! I can not draw because the dimension of your data is not 2!")
		return 1

	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	if k > len(mark):
		print("Sorry! Your k is too large! please contact Zouxy")
		return 1

	# draw all samples
	for i in range(numSamples):
		markIndex = int(clusterAssment[i, 0])
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	# draw the centroids
	for i in range(k):
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)

	plt.show()

def clusterResultCount(clusterAssment):
	result = pd.DataFrame(clusterAssment)
	result.columns = ['class','rate']
	# print(result)

	classed = pd.concat([dna,result],axis=1)
	# print(classed)

	compare = classed[['Kingdom','class']]
	class0 = compare[compare['class']==0]
	class1 = compare[compare['class']==1]
	class2 = compare[compare['class']==2]
	class3 = compare[compare['class']==3]
	class4 = compare[compare['class']==4]
	class5 = compare[compare['class']==5]
	class6 = compare[compare['class']==6]
	class7 = compare[compare['class']==7]

	cluster = [class0,class1,class2,class3,class4,class5,class6,class7]

	verify = []
	for c in cluster:
		vrl = len(c[c['Kingdom']=='vrl'])
		bct = len(c[c['Kingdom']=='bct'])
		pln = len(c[c['Kingdom']=='pln'])
		vrt = len(c[c['Kingdom']=='vrt'])
		inv = len(c[c['Kingdom']=='inv'])
		mam = len(c[c['Kingdom']=='mam'])
		phg = len(c[c['Kingdom']=='phg'])
		rod = len(c[c['Kingdom']=='rod'])
		pri = len(c[c['Kingdom']=='pri'])
		arc = len(c[c['Kingdom']=='arc'])
		plm = len(c[c['Kingdom']=='plm'])
		verify.append([vrl,bct,pln,vrt,inv,mam,phg,rod,pri,arc,plm])
	answer = pd.DataFrame(verify)
	answer.columns = ['vrl','bct','pln','vrt','inv','mam','phg','rod','pri','arc','plm']

	print(answer)


# step 1: load data
print("step 1: load data...")

# step 2: clustering...
print("step 2: clustering...")
dataSet = mat(data.values)
k = 5
# dis_standard = 0 for euclDistance, dis_standard = 1 for manhattanDistance, dis_standard = 1 for cosineDistance
dis_standard = 0
dis_standard = 1
dis_standard = 2
centroids, clusterAssment = kmeans(dataSet, k, dis_standard)

# step 3: show the result
print("step 3: show the result...")
showCluster(dataSet, k, centroids, clusterAssment)

dataSet = mat(data.values)
k = 8
## kmeans + euclDistance
print("kmeans + euclDistance")
dis_standard = 0
centroids, clusterAssment = kmeans(dataSet, k, dis_standard)
clusterResultCount(clusterAssment)

# kmeans + euclDistance
print("\nkmeans + manhattanDistance")
dis_standard = 1
centroids, clusterAssment = kmeans(dataSet, k, dis_standard)
clusterResultCount(clusterAssment)

# kmeans + euclDistance
print("\nkmeans + cosineDistance")
dis_standard = 2
centroids, clusterAssment = kmeans(dataSet, k, dis_standard)
clusterResultCount(clusterAssment)

# kmedians + euclDistance
print("\kmedians + euclDistance")
dis_standard = 0
centroids, clusterAssment = kmedians(dataSet, k, dis_standard)
clusterResultCount(clusterAssment)
