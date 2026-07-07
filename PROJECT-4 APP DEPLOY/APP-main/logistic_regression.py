#!/usr/bin/env python
# coding: utf-8

# <h2 style="color:green" align="center">Predicting if a person would buy life insurnace based on his age using logistic regression</h2>

# Above is a binary logistic regression problem as there are only two possible outcomes (i.e. if person buys insurance or he/she doesn't). 

# In[15]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


df = pd.read_csv("insurance_data.csv")
df.head()


# In[17]:


plt.scatter(df.age,df.bought_insurance,marker='+',color='red')


# In[18]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)


# In[30]:


X_test


# In[31]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[66]:


model.fit(X_train, y_train)


# In[9]:


X_test


# In[39]:


y_predicted = model.predict(X_test)


# In[33]:


model.predict_proba(X_test)


# In[34]:


model.score(X_test,y_test)


# In[40]:


y_predicted


# In[37]:


X_test


# **model.coef_ indicates value of m in y=m*x + b equation**

# In[67]:


model.coef_


# **model.intercept_ indicates value of b in y=m*x + b equation**

# In[68]:


model.intercept_


# **Lets defined sigmoid function now and do the math with hand**

# In[43]:


import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# In[75]:


def prediction_function(age):
    z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)
    return y


# In[76]:


age = 35
prediction_function(age)


# **0.485 is less than 0.5 which means person with 35 age will *not* buy insurance**

# In[77]:


age = 43
prediction_function(age)


# **0.485 is more than 0.5 which means person with 43 will buy the insurance**

# <h2 style="color:purple">Exercise</h2>
# 
# Download employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics. 
# 1. Now do some exploratory data analysis to figure out which variables have direct and clear impact on employee retention (i.e. whether they leave the company or continue to work)
# 2. Plot bar charts showing impact of employee salaries on retention
# 3. Plot bar charts showing corelation between department and employee retention
# 4. Now build logistic regression model using variables that were narrowed down in step 1
# 5. Measure the accuracy of the model
