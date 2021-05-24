import pandas as pd
from numpy import *
from sklearn.cluster import KMeans, Birch
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
data = dna.iloc[:,1:].astype(float)
data1 = dna.iloc[:,1:].astype(float)

def k_means(data):
    k = KMeans(n_clusters = 5).fit_predict(data)

    data.insert(data.shape[1], 'class', k)

    clusterResultCount(data)

def birch(data):
    k = Birch(n_clusters = 5, threshold = 0.001).fit_predict(data)

    data.insert(data.shape[1], 'class', k)

    clusterResultCount(data)

k_means(data)
birch(data1)
