import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv("Salary_Data.csv")
print(dataset)
# 3) we need to split the data based on independent variable(x) and dependent variable(y)

X=dataset.iloc[:,:-1].values

# k=dataset['YearsExperience']
# k        # the above two are same
y=dataset.iloc[:,1:].values


# 5)implement our classifier based on simple linear regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,y)

pickle.dump(regressor,open('slmmodel.pkl','wb'))

with open('slmmodel.pkl', 'rb') as f:
    model=pickle.load(f)