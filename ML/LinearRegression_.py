import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression, PolynomialRegression
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random.randint(0, 100))



regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

old_stdout = sys.stdout

fd = open('file.txt', 'w')
sys.stdout = fd

X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

sys.stdout = old_stdout
fd.close()

plt.plot(y_pred)
plt.plot(y_test)
plt.show()


y_pred_ = regressor.predict(X)
plt.plot(y_pred_, color = 'green', label = 'predicted')
plt.plot(y, color = 'red', label = 'real')
plt.show()

