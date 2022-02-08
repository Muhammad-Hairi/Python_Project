

# Polynomial regression
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

# import dataset

data_set = pd.read_csv('https://trello-attachments.s3.amazonaws.com/6062c6af283e86843a32b025/60c7f8fa059b3a683ff0f11d/127d27d16063cdcdae906079de3a6642/Position_Salaries.csv')
data_set

# Extracting Independent Variable and Dependent Variable
x = data_set.Level.values[:,nm.newaxis]
y = data_set.Salary.values

x

y

# Building The Linear Regression Model
# Fitting the linear regression to the dataset

from sklearn.linear_model import LinearRegression
lin_regs = LinearRegression()
lin_regs.fit(x,y)

# Fitting the Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_regs = PolynomialFeatures(degree=2)
x_poly = poly_regs.fit_transform(x)
x_poly

# when degree is set to 2 the features created will be 1, x1, x2

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Visualizing the result for Linear Regression Model
mtp.scatter(x,y,color='blue')
mtp.plot(x,lin_regs.predict(x),color='red')
mtp.title('Detection Model Using Linear Regression')
mtp.xlabel('Position Levels')
mtp.ylabel('Salary')
mtp.show()

# Visualizing the result for polynomial regression
mtp.scatter(x,y,color='blue')
mtp.plot(x,lin_reg_2.predict(poly_regs.fit_transform(x)),color='red')
mtp.title('Detection Model Using Polynomial Regression')
mtp.xlabel('Position Levels')
mtp.ylabel('Salary')
mtp.show()

# For Degree = 3
poly_regs = PolynomialFeatures(degree=3)
x_poly = poly_regs.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Visualizing the result for polynomial regression when degree = 3
mtp.scatter(x,y,color='blue')
mtp.plot(x,lin_reg_2.predict(poly_regs.fit_transform(x)),color='red')
mtp.title('Detection Model Using Polynomial Regression')
mtp.xlabel('Position Levels')
mtp.ylabel('Salary')
mtp.show()

# For Degree = 4
poly_regs = PolynomialFeatures(degree=4)
x_poly = poly_regs.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Visualizing the result for polynomial regression when degree = 4
mtp.scatter(x,y,color='blue')
mtp.plot(x,lin_reg_2.predict(poly_regs.fit_transform(x)),color='red')
mtp.title('Detection Model Using Polynomial Regression')
mtp.xlabel('Position Levels')
mtp.ylabel('Salary')
mtp.show()

# Predict the final output with Linear Regression Model

lin_pred = lin_regs.predict([[6.5]])
print(lin_pred)

# Predict the final output with Polynomial Regression Model

poly_pred = lin_reg_2.predict(poly_regs.fit_transform([[6.5]]))
print(poly_pred)
