#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[102]:


from sklearn.datasets import load_iris


# In[103]:


iris = load_iris()


# In[104]:


iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


# In[105]:


iris_df.head(n=61)


# In[106]:


# all feature and their datatypes
print(iris_df.dtypes)


# In[107]:


# histogram for each feature in dataset
iris_df.hist(figsize=(7, 7))
plt.show()


# In[108]:


# box plot representation of each feature
sns.boxplot(data=iris_df)
plt.show()


# In[109]:


for col in iris_df.columns:
    Q1 = iris_df[col].quantile(0.25)
    Q3 = iris_df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = iris_df[(iris_df[col] < Q1 - 1.5*IQR) | (iris_df[col] > Q3 + 1.5*IQR)]
    print(f"Outliers in {col}:")
    print(outliers)


# In[ ]:




