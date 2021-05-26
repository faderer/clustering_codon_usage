# 聚类分析

## 1 数据集介绍

我们选取的数据集是[Codon usage Data Set](https://archive.ics.uci.edu/ml/datasets/Codon+usage)，包含了来自不同种群的大量生物样本的DNA密码子出现频率。其中一共具有13028条数据，截取部分数据如下所示。

| Kingdom | DNAtype | SpeciesID | Ncodons |               SpeciesName               |   UUU   |   UUC   |   UUA   | ...  |   UGA   |
| :-----: | :-----: | :-------: | :-----: | :-------------------------------------: | :-----: | :-----: | :-----: | :--: | :-----: |
| **vrl** |    0    |  100217   |  1995   | Epizootic haematopoietic necrosis virus | 0.01654 | 0.01203 |  5E-04  | ...  |    0    |
| **vrl** |    0    |  100220   |  1474   |            Bohle iridovirus             | 0.02714 | 0.01357 | 0.00068 | ...  |    0    |
| **vrl** |    0    |  100755   |  4862   |      Sweet potato leaf curl virus       | 0.01974 | 0.0218  | 0.01357 | ...  | 0.00144 |
| **vrl** |    0    |  100880   |  1915   |      Northern cereal mosaic virus       | 0.01775 | 0.02245 | 0.01619 | ...  |    0    |

每条数据的属性如下：

|     列      |                  属性                   |
| :---------: | :-------------------------------------: |
|  Column 1   |            Kingdom，所属物种            |
|  Column 2   |            DNAtype，DNA类型             |
|  Column 3   |            SpeciesID，物种ID            |
|  Column 4   |           Ncodons，密码子总数           |
|  Column 5   |          SpeciesName，物种名称          |
| Column 6-69 | codon frequencies，某一密码子的出现频率 |

其中，数据集中的频率已是归一化后的结果，无需进一步的处理：
$$
某一密码子的频率=\frac{某一密码子的出现次数}{密码子总数\ Ncodons}
$$
我们将对64种密码子出现的频率作聚类分析，以探索密码子的组成与物种之间的关系。

## 2 数据分析与预处理

首先，我们导入相关工具，并读取数据，检测一共存在哪些物种类别。

```python
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

dna = pd.read_csv('codon_usage.csv')
dna['Kingdom'].value_counts()
```

结果显示如下，数据集中主要的物种为bct(细菌)、vrl(病毒)、pln(植物)、vrt(脊椎动物)、inv(无脊椎动物)等。

```
bct    2919
vrl    2831
pln    2523
vrt    2077
inv    1345
mam     572
phg     220
rod     215
pri     180
arc     126
plm      18
```

由于我们主要需要寻找的是密码子的组成与物种之间的关系，因此“DNAtype“、”SpeciesID“、”Ncodons“、”SpeciesName“等信息是无意义的，需要对其进行剔除。

```python
dna.drop(columns=['DNAtype','SpeciesID','Ncodons','SpeciesName'],inplace=True)
```

为了得到适合的聚类数，我们决定采用交叉验证法。因此在聚类之前，我们将数据等分为10份——其中的9份数据用于训练模型，剩下的1份用于测试集群的质量。对于不同的聚类数k，重复10次，比较不同k的总体质量度量，从而找到最适合的簇数。

```python
from sklearn.utils import shuffle
dna = shuffle(dna)
for i in range(1, 11):
    data = dna[1302*(i-1) : 1302*i - 1]
    data.to_csv("data" + str(i) + ".csv", index=False)
```

## 3 聚类处理

若采用基于密度的方法进行聚类分析，比如DBSCAN算法，由于我们数据的维度过大，因此参数的选取非常困难。比如，DBSCAN用固定参数识别聚类，但当聚类的稀疏程度不同时，相同的判定标准可能会破坏聚类的自然结构，即较稀的聚类会被划分为多个类或密度较大且离得较近的类会被合并成一个聚类。

若采用基于层次的方法进行聚类，比如BIRCH算法，一方面时间复杂度较高，另一方面，阈值threshold难以缺点，经过我们初步的测试，效果并不是非常理想。我们采用Python提供的机器学习库sklearn，对以上数据集进行了简单的聚类分析。

```Python
from sklearn.cluster import Birch

data = dna.iloc[:,1:].astype(float)
k = Birch(n_clusters = 5, threshold = 0.001).fit_predict(data)
```

得到如下的结果，该结果的具体说明将在后面展开。

```python
    vrl   bct   pln   vrt  inv
0   362   937  1042    82  510
1  2167   626   797   153  413
2     1     0     4  1532   34
3    39  1220   116     3   45
4   262   136   564   307  343
```

若采用基于网络的方法，由于参数敏感、无法处理不规则分布的数据、维数灾难等问题，这种算法效率的提高是以聚类结果的精确性为代价的。

而观察我们的数据集，可以发现虽然数据量不大，但是维数众多，难以通过可视化的方式得到密码子频率的分布关系，因此我们最终选择基于划分的方法，即使用k-means算法，进行聚类分析。k-means算法的特点是需要预先设定k值，而我们想要分析的对象是密码子的组成与物种之间的关系，因此物种的种类数量可以作为我们确定k值的依据，并在此基础上进行调整，使用交叉验证法得到最合适的结果。

首先我们去除数据的第一列”Kingdom“。

```Python
data = dna.iloc[:,1:].astype(float)
```

接下来，用于计算欧几里得距离的函数如下所示。

```python
def euclDistance(vector1, vector2):
	return sqrt(sum(power(vector2 - vector1, 2)))
```

用于初始化中心点的函数如下所示。

```python
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids
```

用于进行聚类的函数如下所示。

```python
def kmeans(dataSet, k):
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
 
	print('Congratulations, cluster complete!')
	return centroids, clusterAssment
```

接下来，调用以上的kmeans函数，设定相应的k值，进行聚类分析。

```python
dataSet = mat(data.values)
k = 5
centroids, clusterAssment = kmeans(dataSet, k)
```

以上的kmeans算法为我们自己的实现，为了更好地进行聚类处理，并采用交叉验证的方式选择最合适的k值，我们选择使用sklearn封装的函数，进行聚类分析。

由于数据集中的物种一共有11种类别，而占大多数的物种一共是5种类别，因此，我们将k值的范围取于5-11之间。



