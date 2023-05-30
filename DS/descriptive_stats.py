#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import seaborn as sns


# In[22]:


# Load the iris dataset from seaborn
iris_data = sns.load_dataset('iris')


# In[23]:


# Group data by a categorical variable (e.g., Species) and calculate summary statistics 
# for a numeric variable (e.g., sepal_length)
grouped_stats = iris_data.groupby('species')['sepal_length'].describe()


# In[24]:


# Print the summary statistics
print(grouped_stats)


# In[25]:


# Calculate and print additional summary statistics for each category
grouped_stats_additional = iris_data.groupby('species')['sepal_length'].agg(['mean', 'median', 'min', 'max', 'std'])
print(grouped_stats_additional)


# In[26]:


# Calculate and print percentiles for each species
percentiles = iris_data.groupby('species')['sepal_length'].quantile([0.25, 0.5, 0.75])
print("Percentiles for Iris-setosa:")
print(percentiles.loc['setosa'])
print("Percentiles for Iris-versicolor:")
print(percentiles.loc['versicolor'])
print("Percentiles for Iris-virginica:")
print(percentiles.loc['virginica'])


# In[ ]:




