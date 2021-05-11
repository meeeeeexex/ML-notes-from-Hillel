#k means ++ mrthod
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sk_cluster

objects_sizes_df = pd.read_csv('data\object_sizes.csv')
X = objects_sizes_df[['width','height']]

kmeans_pp_cluster_model = sk_cluster.KMeans(n_clusters=5,n_init=10)
kmeans_pp_cluster_model.fit(X)
kmeans_pp_clasters = kmeans_pp_cluster_model.predict(X)
plt.scatter(X['width'],X['height'],c= kmeans_pp_clasters,cmap='gist_rainbow')#color, grounded on claster
kmeans_pp_centroids = kmeans_pp_cluster_model.cluster_centers_
plt.scatter(kmeans_pp_centroids[:,0],kmeans_pp_centroids[:,1], marker="X", color = 'k',s=100)# s = size
#plt.scatter(X['width'],X['height'])
plt.show()