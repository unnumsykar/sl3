#!/usr/bin/env python
# coding: utf-8

# In[98]:


import seaborn as sns


# In[99]:


titanic_df = sns.load_dataset('titanic')


# In[100]:


sns.boxplot(x='age',y='sex',hue='survived',data=titanic_df,orient='h')


# In[ ]:




