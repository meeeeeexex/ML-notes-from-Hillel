import pandas as pd
import numpy as np
import sklearn.preprocessing as sk_preprocessing
import sklearn.linear_model as sk_linear
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection

muscle_mass_df = pd.read_csv("data/muscle_mass2.csv")
print(muscle_mass_df)
muscle_mass_df.sort_values(by="training_time", inplace=True)
#print(muscle_mass_df)
X = muscle_mass_df[["training_time"]]
y = muscle_mass_df[["muscle_mass"]]
X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(X,y)
plt.scatter(X, y)

polynomial_transformer = sk_preprocessing.PolynomialFeatures(degree=2)
X_transformed_train = polynomial_transformer.fit_transform(X_train)
X_transformed_test = polynomial_transformer.fit_transform(X_test)

#print(X_transformed)

muscle_mass_model = sk_linear.LinearRegression()
muscle_mass_model.fit(X_transformed_train,y_train)

print(muscle_mass_model.intercept_)
modeled_musclr_mass = muscle_mass_model.predict(X_transformed_test)

plt.plot(X, modeled_musclr_mass, color='r')
plt.show()