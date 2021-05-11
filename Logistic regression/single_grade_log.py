import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
df = pd.read_csv('data/single_grade.csv')


#sort the columns
df.sort_values(by=['grade','qualifies'],inplace=True)
X = df[['grade']]
y = df[['qualifies']]
plt.scatter(X,y)

model = sk_linear.LogisticRegression()
model.fit(X,y)
model_qual = model.predict(X)
model_probability = model.predict_proba(X)[:,1]
#write it to our df
df['model probability'] = model_probability
print(df)

plt.plot(X,model_qual,color='k')
plt.plot(X,model_probability,color='g')

#plt.show()

print(model_probability)
#confusion matrix
confusion_matrix= sk_metrics.confusion_matrix(y,model_qual)
print(confusion_matrix)
print("Accuracy:",sk_metrics.accuracy_score(y,model_qual))
print("Error:",1 - sk_metrics.accuracy_score(y,model_qual))
print("Precision:",sk_metrics.precision_score(y,model_qual))
print("Recall:",sk_metrics.recall_score(y,model_qual))
plt.show()