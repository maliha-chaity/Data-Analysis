#!/usr/bin/env python
# coding: utf-8

# ### Linear Regression
# Using the same seattle weather data as last chapter develop a linear regression model

# In[3]:


#import the pakcages that we will need
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('~/data/seattle_weather_1948-2017.csv')

numrows = 25549 # can be as large as 25549

#create an empty dataframe to hold values
regression_df = pd.DataFrame({'intercept':[1] * numrows,
                             'today':[0.0] * numrows,
                             'tomorrow':[0.0] * numrows})
print(regression_df)

#sort columns for convience
seq = ['intercept',
       'today',
       'tomorrow']
regression_df = regression_df.reindex(columns=seq)


# In[4]:


for i in range(0 , numrows):
    tomorrow = df.iloc[i,1]
    today = df.iloc[(i-1),1]
    regression_df.iat[i,2] = tomorrow
    regression_df.iat[i,1] = today

regression_df = regression_df.dropna() #exclude any rows with missing data
print(regression_df)


# In[5]:


#this makes a simple dataframe with a relationship that we can now plot
regression_df.describe


# In[6]:


sns.scatterplot(x='today',y='tomorrow',data=regression_df)


# Creating a basic linear model to best predict these values. Start with a slope and intercept values of 1 and then iterate through gradient descent.

# In[ ]:


def gradientDescent(X, y, param, alpha, num_iters):
    """
       Performs gradient descent to learn param
    """
    for i in range(num_iters):
        y_hat = np.dot(X, param) # notice we use numpy here to efficently calculate the values
        param = param - alpha * np.dot(X.T, y_hat-y)
    return param


# In this fucntion *param* is the initial guess of the values of the linear function and 
# *X* is the vector of data values and y is the realization

# In[ ]:


X = regression_df.iloc[0:200:,0:2]
y = regression_df.iloc[0:200:,2]
param = np.array([1,1])
alpha = 0.0001
num_iters = 1000

solution = gradientDescent(X, y, param, alpha, num_iters)
solution


# In[ ]:


grid = sns.JointGrid(x=regression_df.today,y=regression_df.tomorrow)
grid.plot_joint(plt.scatter, color="g")
plt.plot([0, 5], [0.0811, 1.77], linewidth=2)


# However, building models from scratch is hard! Lucky for us there is a very popular library that trains models for us!

# In[8]:


from sklearn import linear_model
#modify the data to work with this library
x = regression_df.today.values
y = regression_df.tomorrow.values
x = x.reshape(regression_df.shape[0], 1)
y = y.reshape(regression_df.shape[0], 1)


# In[9]:


mymodel = linear_model.LinearRegression().fit(x,y)


# In[10]:


plt.scatter(x, y,  color='black')
plt.plot(x, mymodel.predict(x), color='blue', linewidth=3)
plt.xlim(-0.1,2.1)
plt.ylim(-0.1,2.1)
plt.show()


# In[11]:


# using the r2 (pronounced r squared) value we can get a basic measure of model quality
from sklearn.metrics import r2_score
r2_score(y,mymodel.predict(x))


# We can plot the difference between the predictions and the actual values for a visual estimate of performance. A perfect model would result in this being a straight line with a slope of 1. Notice how the model predicts only lower values, meaning that it tends to under predict the actual amount of rain.

# In[14]:


plt.scatter(mymodel.predict(x), y,  color='black')
plt.xlim(-0.1,2.1) #to keep the same scale as the previous plot
plt.ylim(-0.1,2.1) # same reason as xlim
plt.show()


# ### From this point modify the scikit-learn linear regression method to use two variables.
# Hint. Your x values should have the same number of rows but two columns. You will not be able to plot the line (as it will be 3 dimensional) but you can plot the model predictions agains the actual values.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('~/data/seattle_weather_1948-2017.csv')

numrows = 25547 

###################### adjust this code to add columns here #######################################
# tmean(average of TMAX and TMIN) and today are independent variables. tomorrow is dependent variable.)
regression_df = pd.DataFrame({'tmean':[0.0]*numrows,
                              'today': [0.0]*numrows,
                              'tomorrow':[0.0]*numrows})

#sort columns for convience
seq = ['tmean',
       'today',
       'tomorrow']

regression_df = regression_df.reindex(columns=seq)

for i in range(0 , numrows):
    tmean = (df.iloc[i,2] + df.iloc[i,3]) / 2
    tomorrow = df.iloc[i,1]
    today = df.iloc[(i-1),1]
    regression_df.iat[i,0] = tmean
    regression_df.iat[i,2] = tomorrow
    regression_df.iat[i,1] = today

regression_df = regression_df.dropna() #exclude any rows with missing data
print(regression_df)
#####################################################################################################


# In[3]:


regression_df.describe


# In[4]:


from sklearn import linear_model
x = regression_df.iloc[:, :-1].values
y = regression_df.iloc[:, -1].values
print(x)


# In[5]:


x = x.reshape(regression_df.shape[0], 2)
y = y.reshape(regression_df.shape[0], 1)


# In[6]:


mymodel = linear_model.LinearRegression().fit(x,y)


# In[7]:


#plt.plot(x, mymodel.predict(x), color='blue', linewidth=3)
print("Real values are:")
print(y)


# In[12]:


print("Predicted values are:")
pre = mymodel.predict(x)
print(pre)


# In[17]:


pred = pd.DataFrame({'real_values':[0.0]*len(y),
                     'predicted_values':[0.0]*len(y)})

#sort columns for convience
seq = ['real_values',
       'predicted_values']

pred = pred.reindex(columns=seq)

for i in range(0 , len(y)):
    real_values = y[i]
    predicted_values = pre[i]
    
    pred.iat[i,0] = real_values
    pred.iat[i,1] = predicted_values

pred = pred.dropna() #exclude any rows with missing data
pred.to_csv('~/data/predict_linear.csv', index = True) 


# In[9]:


from sklearn.metrics import r2_score
r2_score(y,mymodel.predict(x))

