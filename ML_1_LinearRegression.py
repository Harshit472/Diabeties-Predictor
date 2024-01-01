#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# In[6]:


from sklearn.datasets import load_diabetes


# In[15]:


x = load_diabetes.data
y = load_diabetes.target

data = pd.DataFrame(x,columns = load_diabetes.feature_names)
data["diabetic"] = y
data.head()


# In[17]:


print(load_diabetes.DESCR)


# In[18]:


print(data.shape)


# In[20]:


data.info()


# In[21]:


data.describe()


# In[25]:


sn.histplot(data['diabetic']);


# In[36]:


fig,ax = plt.subplots()
ax.scatter(x = data['age'],y = data['diabetic'])
plt.ylabel('diabetic',fontsize = 15)
plt.xlabel('age',fontsize = 15)
plt.show()


# In[37]:


fig,ax = plt.subplots()
ax.scatter(x = data['bmi'],y = data['diabetic'])
plt.ylabel('diabetic',fontsize = 15)
plt.xlabel('bmi',fontsize = 15)
plt.show()


# In[38]:


fig,ax = plt.subplots()
ax.scatter(x = data['bp'],y = data['diabetic'])
plt.ylabel('diabetic',fontsize = 15)
plt.xlabel('bp',fontsize = 15)
plt.show()


# In[52]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics

sn.distplot(data['diabetic'] , fit=norm);

(mu, sigma) = norm.fit(data['diabetic'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Diabetes frequency')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(data['diabetic'], plot=plt)
plt.show()


# ## For Reducing the Cost function

# In[60]:


data["diabetic"] = np.log1p(data["diabetic"])

sn.distplot(data['diabetic'] , fit=norm);

(mu, sigma) = norm.fit(data['diabetic'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Diabetes frequency')

fig = plt.figure()
res = stats.probplot(data['diabetic'], plot=plt)
plt.show()


# In[61]:


from sklearn.model_selection import train_test_split
x = data.drop('diabetic',axis=1)
y = data['diabetic']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)


# In[62]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[63]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[64]:


predictions =lr.predict(x_test)
print("Actual Value:",y_test[0])
print("Predicted Value:",predictions[0])


# In[65]:


from sklearn.metrics import mean_squared_error


# In[66]:


mse = mean_squared_error(y_test,predictions)
print(mse)


# In[ ]:




