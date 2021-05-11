#попробовать решить задачу из класса про банк через логистическую регрессию - кол-во лет в банке - зп - данетimp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_tree

loand_df = pd.read_csv('data/loans.csv')
print(loand_df)

def convert_to_numeric (df):
    converted = df.copy()
    converted = converted.replace({'history':{'bad':0,'fair':1,'excellent':2},
                                   'income':{'low':0,'high':1},
                                   'risk':{'low':0,'high':1} })
    return converted
numeric_df = convert_to_numeric(loand_df)
print(numeric_df)

feature_names = loand_df.columns.values[:-1]#делаем чтобы добавить все кроме зависящей пременной- риск
X = numeric_df[feature_names]
y = numeric_df['risk']

loan_decision_tree = sk_tree.DecisionTreeClassifier(criterion='entropy')#default = gini

loan_decision_tree.fit(X,y)

sk_tree.plot_tree(loan_decision_tree,feature_names=feature_names,class_names=['accept credit','decline credit'],
                  filled=True,rounded=True)
#class_names = имена классов которые будут отображаться на графике

plt.show()