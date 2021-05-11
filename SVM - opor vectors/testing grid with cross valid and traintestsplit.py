import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as sk_svm
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import double_grade_svm_utility_Andrii_found

df = pd.read_csv('data/double_grade_reevaluated.csv')
double_grade_svm_utility_Andrii_found.plot_values(df)
print(df)



X = df[['technical_grade','english_grade']]
y = df[['qualifies']]

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X,y)

cv_svn_soft_linear_classifier = sk_svm.SVC(kernel='rbf')
cv_svn_soft_linear_prediction = sk_model_selection.cross_val_predict(cv_svn_soft_linear_classifier,X_train,y_train,cv=4)
cv_conf_matrix = sk_metrics.confusion_matrix(y_train,cv_svn_soft_linear_prediction)
print(cv_conf_matrix)
#на перекрестной валидации посмотрели на качество модели, теперь посмотрим на график, тестируя на всех данных

svn_soft_linear_classifier = sk_svm.SVC(kernel='rbf')
svn_soft_linear_classifier.fit(X_train,y_train)
y_pred = svn_soft_linear_classifier.predict(X_test)
print("Acc:",sk_metrics.accuracy_score(y_test,y_pred))
print("prec:",sk_metrics.precision_score(y_test,y_pred))

parameters = [{'C':[1,10,100,1000],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid_search_cv = sk_model_selection.GridSearchCV(estimator=svn_soft_linear_classifier,
                                                  param_grid=parameters,
                                                  scoring='recall',
                                                  cv=4,
                                                  n_jobs=-1)
grid_search_cv = grid_search_cv.fit(X,y)
acc = grid_search_cv.best_score_
print(acc)
print("\n\n\n")

print(grid_search_cv.best_params_)
print("\n\n\n")


double_grade_svm_utility_Andrii_found.plot_model(svn_soft_linear_classifier)
plt.show()