import pandas as pd
import numpy as np
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import sklearn.preprocessing as sk_preprocessing

f =  open  ('data/text.txt', 'r+')
f.read()
def train_linear_model(X, y):
    linear_model = sk_linear.LinearRegression()
    linear_model.fit(X, y)
    return linear_model

def create_polinomial_model(X_train,y_train,X_test,y_test,n):
    polynomial_transformer = sk_preprocessing.PolynomialFeatures(degree=n)
    X_transformed_train = polynomial_transformer.fit_transform(X_train)
    X_transformed_test = polynomial_transformer.fit_transform(X_test)
    pol_model = train_linear_model(X_transformed_train,y_train)
    res = get_MSE(pol_model,X_transformed_test,y_test)
    return res




def get_MSE(model, X, y_true):
    y_predicted = model.predict(X)
    MSE = sk_metrics.mean_squared_error(y_true, y_predicted)

    return MSE




muscle_mass_df = pd.read_csv("data/muscle_mass.csv")
muscle_mass_df.sort_values(by="training_time", inplace=True)

X = muscle_mass_df[["training_time"]]
y = muscle_mass_df[["muscle_mass"]]

X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X, y, shuffle=True)


safe = {}
for z in range(2,8):
   res = create_polinomial_model(X_train,y_train,X_test,y_test,z)
   safe[z]=res
max = 0,0
def result_print(collection):
    s = ''

    max = 2,collection[2]
    for i in collection:
        if collection[i]<max[1]:
            max = i,collection[i]
        s += "Полином {}-ой степени имеет ошибку {}  \n".format(i,collection[i])
    print(s)
    print(max)
    f.write(str(max[0]))
    f.close()
result_print(safe)
ss = '334347324344326422443323322525524352242364333'
print(len(ss))
mapa= {}
for i in ss:
    mapa[i] = 0
for i in ss:
    mapa[i] += 1
max_in_map = '2'
for i in mapa:
    if mapa[i] > mapa[max_in_map]:
        max_in_map = i
print(mapa)
print(max_in_map)