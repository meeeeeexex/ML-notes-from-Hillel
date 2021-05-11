import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
df = pd.read_csv('data/single_grade.csv')


df.sort_values(by=['grade','qualifies'],inplace=True)
X= df[['grade']]
y = df[['qualifies']]


X_passed = df[df['qualifies']==1]
X_failed = df[df['qualifies']==0]
#print(X_passed)
plt.scatter(X_passed['grade'],X_passed['qualifies'])
plt.scatter(X_failed['grade'],X_failed['qualifies'])


linear_regr = sk_linear.LinearRegression()
lasso_regr = sk_linear.Lasso()
ridge_regr = sk_linear.Ridge()
ridge_regr.fit(X,y)


model_passed = ridge_regr.predict(X)

plt.plot(X,model_passed.round(),color='k')

df['model probability'] = model_passed

confusion_matrix= sk_metrics.confusion_matrix(y,model_passed.round())
print(confusion_matrix)
model_passed = model_passed.round()

print("Accuracy for lin regr:",sk_metrics.accuracy_score(y,model_passed))
print("Error for lin regr:",1 - sk_metrics.accuracy_score(y,model_passed))
print("Precision for lin regr:",sk_metrics.precision_score(y,model_passed))
print("Recall for lin regr:",sk_metrics.recall_score(y,model_passed))
#print(model_passed)




model = sk_linear.LogisticRegression()
model.fit(X,y)
model_qual = model.predict(X)
model_probability = model.predict_proba(X)[:,1]
#write it to our df
df['model probability log_reg'] = model_probability


plt.plot(X,model_qual,color='r')
plt.plot(X,model_probability,color='g')

#plt.show()

#print(model_probability)
#confusion matrix
confusion_matrix_log= sk_metrics.confusion_matrix(y,model_qual)
print("Confusion matrix for log_reg :",confusion_matrix_log)
print("Accuracy for log regr:",sk_metrics.accuracy_score(y,model_qual))
print("Error for log regr:",1 - sk_metrics.accuracy_score(y,model_qual))
print("Precision for log regr:",sk_metrics.precision_score(y,model_qual))
print("Recall for log regr:",sk_metrics.recall_score(y,model_qual))
plt.show()
plt.show()
