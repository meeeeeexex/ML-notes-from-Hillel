import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as sk_svm

import double_grade_svm_utility_Andrii_found

qualities_double_grade_df = pd.read_csv('data/double_grade_small.csv')
double_grade_svm_utility_Andrii_found.plot_values(
    qualities_double_grade_df
)
X = qualities_double_grade_df[['technical_grade','english_grade']]#это так потому что судя про графику, у нас по горизонтали техникал грейд а потом уже инглиш
y = qualities_double_grade_df[['qualifies']]

svm_hard_linear_classfier = sk_svm.SVC(kernel='linear')#svc - support vector classifier
svm_hard_linear_classfier.fit(X,y)

double_grade_svm_utility_Andrii_found.plot_model(svm_hard_linear_classfier)
plt.show()
