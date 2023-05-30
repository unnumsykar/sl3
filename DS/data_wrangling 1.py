#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns


# In[3]:


df = sns.load_dataset('titanic')


# In[4]:


df = pd.DataFrame(df)


# In[5]:


missing_values = df.isnull().sum()
print(missing_values)


# In[6]:


data_description = df.describe()
print(data_description)


# In[7]:


dimensions = df.shape
print(dimensions)


# In[8]:


variable_types = df.dtypes
print(variable_types)


# In[11]:


col_names = df.columns
print(col_names)


# In[12]:


df_encoded = pd.get_dummies(df, columns=['age'])


# In[13]:


df.head()


# In[ ]:




