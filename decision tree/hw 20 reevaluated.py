import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_tree
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.ensemble as sk_ensemble
import sklearn.tree as sk_tree
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_tree
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.ensemble as sk_ensemble
import sklearn.tree as sk_tree
import plotting_with_curve

df = pd.read_csv('data/double_grade_reevaluated.csv')
X = df[['technical_grade','english_grade']]
y = df['qualifies']

X_train,X_test,y_train,y_test = sk_model_selection.train_test_split(X,y)


tree_model = sk_tree.DecisionTreeClassifier()
forest_model = sk_ensemble.RandomForestClassifier()

tree_model.fit(X_train,y_train)
forest_model.fit(X_train,y_train)
plotting_with_curve.plot_model(forest_model,df)

tree_pred = tree_model.predict(X_test)
forest_pred = forest_model.predict(X_test)

print('Tree acc: ',sk_metrics.accuracy_score(y_test,tree_pred))
print('Forest acc: ',sk_metrics.accuracy_score(y_test,forest_pred))

print('Tree prec: ',sk_metrics.precision_score(y_test,tree_pred))
print('Forest prec: ',sk_metrics.precision_score(y_test,forest_pred))
plt.show()