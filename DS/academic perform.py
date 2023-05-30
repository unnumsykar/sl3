#!/usr/bin/env python
# coding: utf-8

# In[15]:


import seaborn as sns


# In[16]:


df = sns.load_dataset('tips')
missing_values = df.isnull().sum()
print(missing_values)


# In[17]:


numeric_vars = df.select_dtypes(include='number')
outliers = df[(numeric_vars - numeric_vars.mean()).abs() > 3 * numeric_vars.std()]
print(outliers)


# In[18]:


# Apply data transformations on a variable (e.g., 'total_bill') to decrease skewness and convert the distribution into a normal distribution:
import numpy as np

df['log_total_bill'] = np.log1p(df['total_bill'])
print(df['log_total_bill'])


# In[ ]:




