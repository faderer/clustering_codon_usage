import pandas as pd
from numpy import *
from sklearn.cluster import KMeans, Birch
from sklearn.utils import shuffle
# from sklearn.metrics import calinski_harabasz_score

def clusterResultCount(data):
    classed = pd.concat([dna["Kingdom"], data["class"]], axis=1)
    compare = classed[['Kingdom','class']]
    class0 = compare[compare['class']==0]
    class1 = compare[compare['class']==1]
    class2 = compare[compare['class']==2]
    class3 = compare[compare['class']==3]
    class4 = compare[compare['class']==4]

    cluster = [class0,class1,class2,class3,class4]

    verify = []
    for c in cluster:
        vrl = len(c[c['Kingdom']=='vrl'])
        bct = len(c[c['Kingdom']=='bct'])
        pln = len(c[c['Kingdom']=='pln'])
        vrt = len(c[c['Kingdom']=='vrt'])
        inv = len(c[c['Kingdom']=='inv'])
        verify.append([vrl,bct,pln,vrt,inv])
    answer = pd.DataFrame(verify)
    answer.columns = ['vrl','bct','pln','vrt','inv']

    print(answer)

dna = pd.read_csv('codon_usage.csv')
dna['Kingdom'].value_counts()

dna.drop(columns=['DNAtype','SpeciesID','Ncodons','SpeciesName'],inplace=True)

# rate = 1
# for i in range(1, 11):
#     print(0.1/rate)
#     data_split = dna.sample(frac=0.1/rate, replace=True, random_state=0, axis=0)
#     data_tmp = dna[~dna.index.isin(data_split.index)]
#     dna = data_tmp
#     data_split.to_csv("data" + str(i) + ".csv", index=False)
#     rate = 1 - 0.1 * i
dna = shuffle(dna)
for i in range(1, 11):
    data = dna[1302*(i-1) : 1302*i - 1]
    data.to_csv("data" + str(i) + ".csv", index=False)

# print(rate)

data = dna.iloc[:,1:].astype(float)
data1 = dna.iloc[:,1:].astype(float)

def k_means(data):
    k = KMeans(n_clusters = 5).fit_predict(data)

    data.insert(data.shape[1], 'class', k)

    clusterResultCount(data)

# def birch(data):
#     k = Birch(n_clusters = 6, threshold = 0.001).fit_predict(data)

#     data.insert(data.shape[1], 'class', k)

#     clusterResultCount(data)


for i in range(1, 11):
    frames = []
    for j in range(1, i):
        frames.append(pd.read_csv('data' + str(j) + '.csv'))
    for j in range(i + 1, 11):
        frames.append(pd.read_csv('data' + str(j) + '.csv'))

    data = pd.concat(frames)

# dna1 = pd.read_csv('data1.csv')
# dna2 = pd.read_csv('codon_usage.csv')
# dna3 = pd.read_csv('codon_usage.csv')
# k_means(data)
# birch(data1)

