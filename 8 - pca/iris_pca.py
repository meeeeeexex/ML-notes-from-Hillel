import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preproc
import sklearn.decomposition as sk_decomposition
import sklearn.linear_model as sk_linear
import sklearn.model_selection as sk_model_selection
import numpy as np

iris_df = pd.read_csv('data/iris.csv').sample(frac=1)
column_names = iris_df.columns.tolist()

X = iris_df[column_names[:-1]]
y = iris_df[column_names[-1]]

stand_scal = sk_preproc.StandardScaler()
X = stand_scal.fit_transform(X)

label_encoder = sk_preproc.LabelEncoder()
y = label_encoder.fit_transform(y)

cv_iris_log_model = sk_linear.LogisticRegression()
cv_iris_model_quality = sk_model_selection.cross_val_score(cv_iris_log_model,X,y,cv=4,scoring='accuracy')
print('Orifinal model acc:')
print(np.mean(cv_iris_model_quality))

pca_2_components = sk_decomposition.PCA(n_components=2)
principal_components = pca_2_components.fit_transform(X)

plt.scatter(principal_components[:,0],principal_components[:,1],c=y,cmap='prism')

cv_iris_2_log_model = sk_linear.LogisticRegression()
cv_iris_2_model_quality = sk_model_selection.cross_val_score(cv_iris_2_log_model,principal_components,y,cv=4,scoring='accuracy')
print('model2 w  pca:')
print(np.mean(cv_iris_2_model_quality))

pca_all_components = sk_decomposition.PCA()
pca_all_components.fit(X)

print('Explained variance raitio:')
print(pca_all_components.explained_variance_ratio_)# изобразим графики рещультата так как это показывает влияние плоскости на результатат
plt.figure()
components = list(range(1,pca_all_components.n_components_ + 1))
plt.plot(components,np.cumsum(pca_all_components.explained_variance_ratio_),marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained cariance')
plt.show()

