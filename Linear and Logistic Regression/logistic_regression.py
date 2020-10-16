#!/usr/bin/env python
# coding: utf-8

# ### Logistic Regression
# Using the same seattle weather data as last chapter develop a linear regression model

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('~/data/seattle_weather_1948-2017.csv')

numrows = 25549 # can be as large as 25549

#create an empty dataframe to hold values
regression_df = pd.DataFrame({'today':[0.0] * numrows,
                             'tomorrow':[True] * numrows}) #initalize as boolean

#sort columns for convience
seq = ['today',
       'tomorrow']

regression_df = regression_df.reindex(columns=seq)


# In[2]:


#populate the regression data frame with values from the pandas dataframe
for i in range(0 , numrows):
    tomorrow = df.iloc[i,1]
    today = df.iloc[(i-1),1]
    regression_df.iat[i,1] = tomorrow
    regression_df.iat[i,0] = today

regression_df = regression_df.dropna() #exclude any rows with missing data


# In[3]:


regression_df.head(20)


# In[21]:


from sklearn import linear_model
#modify the data to work with this model
x = regression_df.today.values
y = regression_df.tomorrow.values
x = x.reshape(regression_df.shape[0], 1)
#note that we did not need to reshape the y values as we did with linear regression


# In[19]:


clf = linear_model.LogisticRegression(solver='lbfgs').fit(x, y)


# In[6]:


#we can calculate the accuarcy using the score method
score = clf.score(x,y)
print(score)


# In[7]:


from sklearn import metrics
#we can also make a simple confusion matrix
predictions = clf.predict(x)
cm = metrics.confusion_matrix(y, predictions)
print(cm)


# In[8]:


#Here is a bit nicer matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# ### From this point modify the scikit-learn logistic regression method to use two variables.
# Hint. Your x values should have the same number of rows but two columns. You will not be able to plot the line (as it will be 3 dimensional) but you can plot the model predictions agains the actual values.

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('~/data/seattle_weather_1948-2017.csv')

numrows = 25547 

###################### adjust this code to add columns here #######################################
# independent variables are prcp, tmax and tmin. Dependent variable is tomorrow.
regression_df = pd.DataFrame({'prcp':[0.0]*numrows,
                              'tmax':[0.0]*numrows,
                              'tmin':[0.0]*numrows,
                              'tomorrow':[0.0]*numrows })

#sort columns for convience
seq = ['prcp',
       'tmax',
       'tmin',
       'tomorrow']

regression_df = regression_df.reindex(columns=seq)

for i in range(0 , numrows):
    prcp = df.iloc[i,1]
    tmax = df.iloc[i,2]
    tmin = df.iloc[i,3]
    tomorrow = df.iloc[i,4]
    
    regression_df.iat[i,0] = prcp
    regression_df.iat[i,1] = tmax
    regression_df.iat[i,2] = tmin
    regression_df.iat[i,3] = tomorrow

regression_df = regression_df.dropna() #exclude any rows with missing data
print(regression_df)
#####################################################################################################


# In[6]:


regression_df.describe


# In[7]:


from sklearn import linear_model
x = regression_df.iloc[:, :-1].values
y = regression_df.iloc[:, -1].values
print(x)


# In[8]:


x = x.reshape(regression_df.shape[0], 3)
#y = y.reshape(regression_df.shape[0], 1)


# In[9]:


mymodel = linear_model.LogisticRegression(solver='lbfgs').fit(x,y)


# In[10]:


print("Real values are:")
print(y)


# In[20]:


print("Predicted values are:")
pre = mymodel.predict(x)
print(pre)


# In[21]:


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
pred.to_csv('~/data/predict_logistic.csv', index = True) 


# In[18]:


# Calculating accuracy with score method
score = mymodel.score(x,y)
print(score)


# In[13]:


# Making confusion matrix
from sklearn import metrics
predictions = mymodel.predict(x)
cm = metrics.confusion_matrix(y, predictions)
print(cm)


# In[14]:


# Finding precision
pr = metrics.precision_score(y, predictions)
print(pr)


# In[15]:


# Finding recall
rc = metrics.recall_score(y, predictions)
print(rc)


# In[16]:


# Finding sum of squared error(SSE)
mse = metrics.mean_squared_error(y, predictions)
sse = mse * numrows
print(sse)

