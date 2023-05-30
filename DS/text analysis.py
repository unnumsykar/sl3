#!/usr/bin/env python
# coding: utf-8

# In[83]:


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[84]:


document = "This is an example document for tokenization. , This is an example document for POS tagging ,stemming"


# In[85]:


#Document Preprocessing
tokens = word_tokenize(document)
print(tokens)


# In[86]:


#POS Tagging
pos_tags = pos_tag(tokens)
print(pos_tags)


# In[87]:


#Stop Words Removal
stop_words = set(stopwords.words('english'))


# In[88]:


filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print(filtered_tokens)


# In[89]:


#Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print(stemmed_tokens)


# In[90]:


#Lemmatization
lemmatizer = WordNetLemmatizer()


# In[91]:


lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print(lemmatized_tokens)


# In[ ]:




