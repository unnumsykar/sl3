#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# In[54]:


# Load the iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target


# In[55]:


# Split the dataset into features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']


# In[56]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[57]:


# Create a Gaussian Naïve Bayes model
model = GaussianNB()


# In[58]:


# Train the model
model.fit(X_train, y_train)


# In[59]:


# Predict on the testing set
y_pred = model.predict(X_test)


# In[60]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# In[61]:


# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[62]:


# Compute the error rate
error_rate = 1 - accuracy
print("Error Rate:", error_rate)


# In[63]:


# Compute the precision
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)


# In[64]:


# Compute the recall
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)


# In[ ]:




