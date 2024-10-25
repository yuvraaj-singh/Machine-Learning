# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('Independent Variables/ Feature Variables:',x)
print('Dependent Variables/ Target Variables:',y)

# Taking care of missing data
from sklearn.impute import SimpleImputer;
# Print the number of missing entries in each column
missing_values = dataset.isnull().sum()
print('Missing Values in each column: \n',missing_values)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Encoding the Independent Variable
# OneHotEncoder is used to convert the categorical data into binary format
# ColumnTransformer is used to specify which columns to be transformed
# remainder='passthrough' is used to keep the other columns as it is
# [0] is used to specify the column to be transformed
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
# LabelEncoder is used to convert the categorical data into numerical format
# Here, we have only one column, so we can directly use fit_transform
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print('X_train:',X_train)
print('X_test:',X_test)
print('y_train:',y_train)
print('y_test:',y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
print('X_train after scaling: \n',X_train)

X_test[:, 3:] = sc.transform(X_test[:, 3:])
print('X_test after scaling:',X_test)

