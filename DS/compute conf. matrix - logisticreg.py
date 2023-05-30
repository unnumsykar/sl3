#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# In[43]:


# Load the breast cancer dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target


# In[44]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[45]:


# Create a logistic regression model
model = LogisticRegression()


# In[46]:


# Train the model
model.fit(X_train, y_train)


# In[47]:


# Predict on the testing set
y_pred = model.predict(X_test)


# In[48]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# In[49]:


# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[50]:


# Compute the error rate
error_rate = 1 - accuracy
print("Error Rate:", error_rate)


# In[51]:


# Compute the precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)


# In[52]:


# Compute the recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)


# In[ ]:




