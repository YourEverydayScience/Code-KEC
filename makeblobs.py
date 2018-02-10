
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import pandas as pd
import os
from sklearn import cluster
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np

def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = cluster.KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = cluster.KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1,
            resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

from sklearn.datasets.samples_generator import make_blobs

X, ind = make_blobs(n_samples=1000, centers=3, n_features=2,random_state=0, cluster_std=0.5)
x = X[:,0]
y = X[:,1]


ks = range(1, 20)

# Run KMeans with a batch of 10000 data points for the desired iterations with data X
#Kmeans = [cluster.KMeans(n_clusters=i,tol=1e-4,random_state=None).fit(X) for i in ks]

from sklearn import cluster

k, gapdf = optimalK(X,nrefs=3,maxClusters=15)

print(gapdf)

#BIC = [compute_bic(kmeansi, X) for kmeansi in Kmeans]
#BIC = [compute_bic(kmeansi, X) for kmeansi in Kmeans]
#for i in range(0, len(x)):
#    plt.scatter(x, y,c=ind)

#print(BIC)

kmeans = cluster.KMeans(n_clusters=3).fit(X)
plt.scatter(x, y,c=kmeans.labels_)
plt.title('Three Normally Distributed Clusters')
plt.xlabel("X Position")
plt.ylabel('Y Position')
plt.show()
#plt.plot(gapdf['clusterCount'], gapdf['gap'])
# plt.title('Implementing the Gap Statistic on Three Clusters')
# plt.xlabel("Number of Clusters 'k'")
# plt.ylabel('Gap Statistic Score')
# plt.show()



