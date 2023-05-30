#!/usr/bin/env python
# coding: utf-8

# In[93]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[94]:


# Load the 'titanic' dataset from Seaborn
titanic = sns.load_dataset('titanic')


# In[96]:


# Convert 'fare' column to numeric
titanic['fare'] = pd.to_numeric(titanic['fare'], errors='coerce')


# In[97]:


# Plot a histogram of the ticket prices
sns.histplot(data=titanic, x="fare", kde=True)
plt.xlabel("Ticket Price")
plt.ylabel("Frequency")
plt.title("Distribution of Ticket Prices")
plt.show()


# In[ ]:




