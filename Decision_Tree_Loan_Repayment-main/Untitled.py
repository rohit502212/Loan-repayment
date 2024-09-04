#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[52]:


balance_data = pd.read_csv('Decision_Tree_ Dataset.csv')


# In[53]:


print("DataSet Length")
len(balance_data)


# In[54]:


print("Dataset Shape")


# In[55]:


balance_data.shape


# In[56]:


print("Full DataSet::")
balance_data.head()


# ## Making inputs and test outputs 

# In[58]:


X = balance_data.values[: , 0:5]
Y = balance_data.values[: , 5]


# In[59]:


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[60]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


# We have trained our model

# In[61]:


y_pred = clf_entropy.predict(X_test)
y_pred


# In[63]:


print("Accuracy ::")
print(accuracy_score(y_pred , y_test))


# In[ ]:




