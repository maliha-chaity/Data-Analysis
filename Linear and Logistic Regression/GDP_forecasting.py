#!/usr/bin/env python
# coding: utf-8

# In[2]:


# train the modifiedGDP file and predict GDPs of 2018 for all the countries using linear regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#import warnings
#warnings.simplefilter("ignore")


# In[3]:


df = pd.read_csv('~/data/modifiedGDP.csv')


# In[4]:


df.drop(df[df['Country Name'] == "Lower middle income"].index, inplace = True)
df.drop(df[df['Country Name'] == "Low & middle income"].index, inplace = True)
df.drop(df[df['Country Name'] == "Late-demographic dividend"].index, inplace = True)
df.drop(df[df['Country Name'] == "Middle income"].index, inplace = True)
df.drop(df[df['Country Name'] == "Pre-demographic dividend"].index, inplace = True)
df.drop(df[df['Country Name'] == "Post-demographic dividend"].index, inplace = True)
df.drop(df[df['Country Name'] == "Upper middle income"].index, inplace = True)
df.drop(df[df['Country Name'] == "World"].index, inplace = True)
df.drop(df[df['Country Name'] == "Early-demographic dividend"].index, inplace = True)


# In[5]:


print(df)


# In[6]:


df = df.replace(to_replace ="Unnamed: 62", value = 2018)
df


# In[7]:


df = df.drop(['Country Code'], axis = 1)
df


# In[8]:


df['Encoded'] = ''
df


# In[9]:


df = df[['Country Name', 'Encoded', 'Year', 'GDP']] 
df


# In[10]:


from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
df['Encoded'] = number.fit_transform(df['Country Name'].astype('str'))
df


# In[11]:


df = df.replace(np.nan, 0)
df


# In[12]:


from sklearn import linear_model
x = df.iloc[:, [1,2]].values
y = df.iloc[:, -1].values
print(x)


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 255, random_state = 0, shuffle = False)


# In[14]:


print(x_train)


# In[15]:


print(y_train)


# In[16]:


print(x_test)


# In[17]:


mymodel = linear_model.LinearRegression().fit(x,y)


# In[18]:


print("Predicted values are:")
pre = mymodel.predict(x_test)
print(pre)


# In[19]:


df1 = df.tail(255)
df1


# In[20]:


df = df.replace(0, np.nan)
df


# In[21]:


df = df.dropna()
df


# In[22]:


#k = 15313
for i in range(0, len(pre)):
    val = pre[i]
#    k = k + 1
    
    df1.iat[i,3] = val
#    k = k + 1

df1.to_csv('~/data/GDP_prediction.csv', index = True) 


# In[23]:


df


# In[24]:


frames = [df, df1]
 
res1 = pd.concat(frames)
res1


# In[25]:


res1 = res1.drop(['Encoded'], axis = 1)
res1


# In[26]:


res1.to_csv('~/data/new_GDP.csv', index = True)


# In[27]:


cn_summary = res1.loc[res1['Country Name'] == 'Bangladesh']


# In[28]:


cn_summary.describe()


# In[29]:


cn_summary.head()


# In[30]:


cn_summary.plot.line(x = 'Year', y = 'GDP')

