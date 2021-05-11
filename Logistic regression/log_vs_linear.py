import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection

df = pd.read_csv('data/linear_vs_logistic.csv')
X= df[['grade']]
y = df[['qualifies']]
plt.scatter(X,y)
linear_regression = sk_linear.LinearRegression()
linear_regression.fit(X,y)
prediction_lin = linear_regression.predict(X)


logistic_regression = sk_linear.LogisticRegression()
logistic_regression.fit(X,y)
prediction_log = logistic_regression.predict(X)

df['Linear model'] = prediction_lin.round()
df['Logistic model'] = prediction_log
plt.plot(X,prediction_lin.round(),color='k')
plt.plot(X,prediction_log,color='r')
prediction_lin = prediction_lin.round()


print(df)


confusion_matrix_lin = sk_metrics.confusion_matrix(y,prediction_lin)
confusion_matrix_log = sk_metrics.confusion_matrix(y,prediction_log)

print("Матрица  ЛИНЕЙНОЙ {}".format(confusion_matrix_lin))
print("Матрица  ЛОГИСТЕЧЕСКОЙ {}".format(confusion_matrix_log))
print("Accuracy for linear:",sk_metrics.accuracy_score(y,prediction_lin))
print("Error for linear:",1 - sk_metrics.accuracy_score(y,prediction_lin))
print("Precision for linear:",sk_metrics.precision_score(y,prediction_lin))
print("Recall for linear:",sk_metrics.recall_score(y,prediction_lin))

print('\n\n\n')

print("Accuracy for logistic:",sk_metrics.accuracy_score(y,prediction_log))
print("Error for logistic:",1 - sk_metrics.accuracy_score(y,prediction_log))
print("Precision for logistic:",sk_metrics.precision_score(y,prediction_log))
print("Recall for logistic:",sk_metrics.recall_score(y,prediction_log))
plt.legend(['Linear','Logistic'])

plt.show()
