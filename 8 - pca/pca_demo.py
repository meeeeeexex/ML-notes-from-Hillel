import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preproc
import sklearn.decomposition as sk_decomposition
import numpy as np

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=((12,6)))
demo_df = pd.read_csv('data/pca_demo_data.csv')

standard_scal = sk_preproc.StandardScaler()
X_standard = standard_scal.fit_transform(demo_df[['x','y']])

#plt.scatter(demo[x],semo[y])
# comare with
ax1.scatter(X_standard[:,0],X_standard[:,1])

#######
pca_trans = sk_decomposition.PCA(n_components=2)
principal_comp =pca_trans.fit_transform(X_standard)
#ax2.scatter(principal_comp[:,0],np.zeros(len(X_standard)))#1-d space depends on n_components
ax2.scatter(principal_comp[:,0],principal_comp[:,1])#2d scpace

print(pca_trans.components_)

plt.show()