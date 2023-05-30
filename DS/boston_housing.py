#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[35]:


# Load the Boston Housing dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['target'] = boston.target


# In[36]:


# Split the dataset into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


# Create a linear regression model
model = LinearRegression()


# In[38]:


# Train the model
model.fit(X_train, y_train)


# In[39]:


# Predict on the testing set
y_pred = model.predict(X_test)


# In[40]:


# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[41]:


# Calculate the coefficient of determination (R-squared)
r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)


# In[ ]:




