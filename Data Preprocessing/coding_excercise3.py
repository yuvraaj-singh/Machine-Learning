# Importing the necessary libraries
import numpy as np;
import pandas as pd;
from sklearn.impute import SimpleImputer;


# Load the dataset
df = pd.read_csv('titanic.csv');


# Identify the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = ['Sex','Embarked','Pclass']

# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')


# Apply the fit_transform method on the instance of ColumnTransformer
X = ct.fit_transform(df)

# Convert the output into a NumPy array
X = np.array(X)

# Use LabelEncoder to encode binary categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['Survived'])

# Print the updated matrix of features and the dependent variable vector
print(X)
print(y)