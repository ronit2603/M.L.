
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel(r'C:\Users\LENOVO\Desktop\M.L\Advertise.xlsx')
dataset.head(10)


# In[117]:


X=dataset[['TV','radio','newspaper']]
y=dataset.sales

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)

print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))
print(lr.score(X,y))

