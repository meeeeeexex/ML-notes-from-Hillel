import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_tree
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import sklearn.ensemble as sk_ensemble
import sklearn.tree as sk_tree
import graphviz
import plotting_with_curve

f =  open  ('data/text.txt', 'r+')
f.read()
plt.figure(figsize=(12,8))

double_grade_df = pd.read_csv('data/double_grade.csv')
feature_names = double_grade_df.columns.values[:-1]
X = double_grade_df[feature_names]
y = double_grade_df['qualifies']
def tree(X,y):
    acc_sum, prec_sum = 0, 0
    def create(X,y):
        X_train , X_test, y_train, y_test = sk_model_selection.train_test_split(X,y)

        grade_decision_tree = sk_tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
        grade_decision_tree.fit(X,y)

        grade_with_spliting = sk_tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
        grade_with_spliting.fit(X_train,y_train)
        tree_y_pred_split = grade_with_spliting.predict(X_test)

        # s  = 'Acc: '+str(sk_metrics.accuracy_score(y_test,tree_y_pred_split))+'\t'  + 'Precision: '+ \
        #      str(sk_metrics.precision_score(y_test,tree_y_pred_split))+'\n'
        acc = sk_metrics.accuracy_score(y_test,tree_y_pred_split)
        prec = sk_metrics.precision_score(y_test,tree_y_pred_split)
        print('Acc: ',sk_metrics.accuracy_score(y_test,tree_y_pred_split))
        print('Precision: ',sk_metrics.precision_score(y_test,tree_y_pred_split))
        print('\n')

        sk_tree.plot_tree(grade_decision_tree,feature_names=feature_names,class_names=['Unqualified','Qualified'],
                          filled=True)
        tree_y_pred = grade_decision_tree.predict(X)

        print('Acc: ',sk_metrics.accuracy_score(y,tree_y_pred))

        print('Precision: ',sk_metrics.precision_score(y,tree_y_pred))
        # tree_y_pred = grade_decision_tree.predict(X_test)
        #
        # print('Acc: ',sk_metrics.accuracy_score(y_test,tree_y_pred))
        # print('Precision: ',sk_metrics.precision_score(y_test,tree_y_pred))
        tree_conf_matrix = sk_metrics.confusion_matrix(y,tree_y_pred)
        print(tree_conf_matrix)
        return acc,prec
        #f.write(s)


    for i in range(50):
       acc_tmp,prec_tmp = create(X,y)
       acc_sum+=acc_tmp
       prec_sum += prec_tmp
    acc_average = acc_sum/50
    prec_avg = prec_sum/50
    print(acc_average)
    print(prec_avg)
    s = 'accuracy with depth 3 :'+ str(acc_average)+' precision: '+ str(prec_avg)+'\n'
    f.write(s)
    f.close()
def forest(X,y):
    acc_sum, prec_sum = 0, 0

    def create(X, y):
        X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y)

        grade_forest = sk_ensemble.RandomForestClassifier(n_jobs=-1)
        grade_forest.fit(X_train,y_train)
        y_predicted = grade_forest.predict(X_test)

        acc = sk_metrics.accuracy_score(y_test,y_predicted)
        prec = sk_metrics.precision_score(y_test,y_predicted)

        print('Acc: ', acc)

        print('Precision: ', prec,'\n')

        return acc, prec
    for i in range(50):
        acc_tmp, prec_tmp = create(X,y)
        acc_sum += acc_tmp
        prec_sum += prec_tmp
    acc_avg = acc_sum/50
    prec_avg = prec_sum/50
    print('-----------------------------------\n')
    print('Acc average: ', acc_avg)

    print('Precision average: ', prec_avg, '\n')
#forest(X,y)

plt.show()