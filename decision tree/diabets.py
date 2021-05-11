import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_tree
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.ensemble as sk_ensemble
diabets_df = pd.read_csv('data/pima-indians-diabetes.csv')
column_names = diabets_df.columns.values

X = diabets_df[column_names[:-1]]
y = diabets_df[column_names[-1]]

X_train , X_test, y_train, y_test = sk_model_selection.train_test_split(X,y)


print('Decision tree :')
diabets_tree_model= sk_tree.DecisionTreeClassifier(max_depth=4)

diabets_tree_model.fit(X_train,y_train)

tree_y_pred = diabets_tree_model.predict(X_test)

print('Acc: ',sk_metrics.accuracy_score(y_test,tree_y_pred))
print('Precision: ',sk_metrics.precision_score(y_test,tree_y_pred))
tree_conf_matrix = sk_metrics.confusion_matrix(y_test,tree_y_pred)
print(tree_conf_matrix)

#sk_tree.plot_tree(diabets_tree_model,feature_names=column_names,class_names=['0','1'],filled=True,rounded=True)
print('\n\n')


print('Random forest model: ')
diabets_forest_model = sk_ensemble.RandomForestClassifier(n_jobs=-1)

diabets_forest_model.fit(X_train,y_train)
forest_y_pred = diabets_forest_model.predict(X_test)

print('Acc: ',sk_metrics.accuracy_score(y_test,forest_y_pred))
print('Precision: ',sk_metrics.precision_score(y_test,forest_y_pred))
forest_conf_matrix = sk_metrics.confusion_matrix(y_test,forest_y_pred)

print(forest_conf_matrix)

plt.show()