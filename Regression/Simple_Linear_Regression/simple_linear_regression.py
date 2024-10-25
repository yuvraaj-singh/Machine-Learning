# Importing libraries
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
# print(X)
y = dataset.iloc[:, 1].values
# print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split;
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the simple regression model on the Training set
from sklearn.linear_model import LinearRegression;
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))



# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)

# Therefore, the equation of our simple linear regression model is:
# Salary = 9312.57512673 * YearsExperience + 26780.09915062818
# Important note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values.
# Methods instead return a more complex object, which has methods itself. That's why when we call "predict()" for example, we call it as a method with the parenthesis. But when we call "coef_" and "intercept_", we're just accessing a simple attribute, not calling a method.
