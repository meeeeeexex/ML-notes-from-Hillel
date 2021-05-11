import pandas as pd
import numpy as np
import sklearn.linear_model as sk_linear
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection


def model_to_string(model, labels, precision=6):
    model_str = "{} = ".format(labels[-1])
    for z in range(len(labels) - 1):
        model_str += "{} * {} + ".format(round(model.coef_.flatten()[z], precision), labels[z])
    model_str += "{}".format(round(model.intercept_[0]), precision)
    return model_str


advertising_data = pd.read_csv("data/advertising.csv", index_col=0)
#print(advertising_data)

ad_data = advertising_data[["TV", "radio", "newspaper"]]
sales_data = advertising_data[["sales"]]

labels = advertising_data.columns.values

linear_regr = sk_linear.LinearRegression()
lasso_regr = sk_linear.Lasso()
ridge_regr = sk_linear.Ridge()

linear_regr.fit(ad_data,sales_data)
lasso_regr.fit(ad_data,sales_data)
ridge_regr.fit(ad_data,sales_data)



print("Linear regr.")
print(model_to_string(linear_regr,labels))


print("L1 regr.")
print(model_to_string(lasso_regr,labels))


print("L2 regr.")
print(model_to_string(ridge_regr,labels))

