import pandas as pd
import numpy as np
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import sklearn.preprocessing as sk_preprocessing
import matplotlib.pyplot as plt
def train_linear_model(X, y):
    linear_model = sk_linear.LinearRegression()
    linear_model.fit(X, y)
    return linear_model

def create_polinomial_model(X,y,n):
    polynomial_transformer = sk_preprocessing.PolynomialFeatures(degree=n)
    X_transformed = polynomial_transformer.fit_transform(X)
    pol_model = train_linear_model(X_transformed,y)
    res = get_MSE(pol_model,X_transformed,y)
    return res




def get_MSE(model, X, y_true):
    y_predicted = model.predict(X)
    MSE = sk_metrics.mean_squared_error(y_true, y_predicted)

    return (MSE,y_predicted)




muscle_mass_df = pd.read_csv("data/muscle_mass.csv")
muscle_mass_df.sort_values(by="training_time", inplace=True)

X = muscle_mass_df[["training_time"]]
y = muscle_mass_df[["muscle_mass"]]



res1 = create_polinomial_model(X,y,15)
print(res1[0])


plt.scatter(X, y)
plt.plot(X,res1[1], color='r')
plt.show()
