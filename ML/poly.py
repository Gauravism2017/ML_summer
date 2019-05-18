import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
import random
import numpy as np
import sys
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")
data_1 = pd.read_csv("50_Startups.csv")

dummies = pd.get_dummies(data_1.State)
data_1 = pd.concat([data_1, dummies], axis=1)
data_1.drop(['State'], axis=1, inplace=True)

y = data_1['Profit'].values
data_1.drop(['Profit'], axis=1, inplace=True)
X = data_1.iloc[:,:].values

sc_X = StandardScaler()
sc_1 = sc_X.fit(X)
X = sc_1.transform(X)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random.randint(0, 100))

poly = PolynomialFeatures(degree = 4)
x_poly = poly.fit(X_train)
X_poly = x_poly.transform(X_train)
#X_train_.fit(X_train, y)
print(X_train.shape)
#poly.fit(X_poly , y_train)
regressor = LinearRegression(n_jobs = 4)
regressor.fit(X_poly, y_train)


X_test_poly = x_poly.transform(X_test)
y_pred = regressor.predict(X_test_poly)


plt.plot(y_test, color = 'green')
plt.plot(y_pred, color = 'red')
plt.show()