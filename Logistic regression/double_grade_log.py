import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
df = pd.read_csv('data/double_grade.csv')
#sort the columns
#df.sort_values(by=['technical_grade','english_grade','qualifies'],inplace=True)
X = df[['technical_grade','english_grade']]
y = df[['qualifies']]
# paint scatters
qualified_candidates = df[df['qualifies']==1]
unqualified_candidates = df[df['qualifies']==0]
plt.scatter(qualified_candidates['technical_grade'],qualified_candidates['english_grade'],color = 'g')
plt.scatter(unqualified_candidates['technical_grade'],unqualified_candidates['english_grade'],color = 'r')

plt.xlabel('Technical grade')
plt.ylabel('English grade')
k_folds = 4
crossvalid_qual_model = sk_linear.LogisticRegression()

cv_model_qual = sk_model_selection.cross_val_score(crossvalid_qual_model, X, y, cv=k_folds, scoring='accuracy')
print(cv_model_qual)

cv_pred = sk_model_selection.cross_val_predict(crossvalid_qual_model,X,y,cv=k_folds)
cv_conf_matrix = sk_metrics.confusion_matrix(y,cv_pred)
print(cv_conf_matrix)

qualification_model = sk_linear.LogisticRegression()
qualification_model.fit(X,y)


model_probability = qualification_model.predict_proba(X)[:,1]
df['model probability'] = model_probability


pd.set_option('display.max_rows',None)
print(df.sort_values(by='model probability'))

print(qualification_model.coef_)
print(qualification_model.intercept_)

#lets print plot fro this task

k1,k2 = qualification_model.coef_.flatten()
b = qualification_model.intercept_[0]

x_boundary =  [df['technical_grade'].min(),df['technical_grade'].max()]
y_boundary = [-(k1*x+b)/k2 for x in x_boundary]

plt.plot(x_boundary,y_boundary,color='b')

plt.show()

'''
plt.plot(X,model_qual,color='k')
plt.plot(X,model_probability,color='g')

#plt.show()


#confusion matrix
confusion_matrix= sk_metrics.confusion_matrix(y,model_qual)
print(confusion_matrix)
print("Accuracy:",sk_metrics.accuracy_score(y,model_qual))
print("Error:",1 - sk_metrics.accuracy_score(y,model_qual))
print("Precision:",sk_metrics.precision_score(y,model_qual))
print("Recall:",sk_metrics.recall_score(y,model_qual))
plt.show()
'''