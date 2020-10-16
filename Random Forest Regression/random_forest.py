#!/usr/bin/env python
# coding: utf-8

# ### Random Forest
# Using the same seattle weather data as last chapter develop a linear regression model

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import ensemble
from sklearn import metrics

df = pd.read_csv('~/data/seattle_weather_1948-2017.csv')

numrows = 25549 

randomforest_df = pd.DataFrame({'today':[0.0]*numrows,
                             'tomorrow':[True]*numrows})

#sort columns for convience
seq = ['today',
       'tomorrow']

randomforest_df = randomforest_df.reindex(columns=seq)

for i in range(0 , numrows):
    tomorrow = df.iloc[i,1]
    today = df.iloc[(i-1),1]
    randomforest_df.iat[i,1] = tomorrow
    randomforest_df.iat[i,0] = today

randomforest_df = randomforest_df.dropna() #exclude any rows with missing data
print(randomforest_df)


# In[3]:


#modify the data to work with this model
x = randomforest_df.today.values
y = randomforest_df.tomorrow.values
x = x.reshape(randomforest_df.shape[0], 1)
#note that we did not need to reshape the y values as we did with linear regression


# In[4]:


clf = ensemble.RandomForestClassifier(n_estimators=100).fit(x, y)


# In[5]:


#we can calculate the accuarcy using score
score = clf.score(x,y)
print(score)


# In[6]:


#Conf Matrix

#we can also make a simple confusion matrix
predictions = clf.predict(x)
cm = metrics.confusion_matrix(y, predictions)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# ### From this point modify the scikit-learn random forest method to use two or more variables.
# Hint. Your x values should have the same number of rows but two columns. You will not be able to plot the line (as it will be 3 dimensional) but you can plot the model predictions agains the actual values.

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from sklearn import ensemble
from sklearn import metrics

df = pd.read_csv('~/data/seattle_weather_1948-2017.csv')

numrows = 25549 

###################### adjust this code to add columns here #######################################
# independent variables are prcp, tmax and tmin. Dependent variable is tomorrow.
regression_df = pd.DataFrame({'prcp':[0.0]*numrows,
                              'tmax':[0.0]*numrows,
                              'tmin':[0.0]*numrows,
                              'tomorrow':[0.0]*numrows })

#sort columns for convience
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


# In[8]:


regression_df.describe


# In[9]:


x = regression_df.iloc[:, :-1].values
y = regression_df.iloc[:, -1].values
print(x)


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[12]:


print(x_train)


# In[13]:


print(y_train)


# In[14]:


print(x_test)


# In[15]:


print(y_test)


# In[18]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf.fit(x_train, y_train)


# In[19]:


# Real values for the training set of x_train
print("Real values are:")
print(y_train)


# In[25]:


# Predicted values for the training set of x_train
p1 = rf.predict(x_train)
print("Predicted values are:")
print(p1)


# In[34]:


pred_train = pd.DataFrame({'real_values':[0.0]*len(y_train),
                     'predicted_values':[0.0]*len(y_train)})

#sort columns for convience
seq = ['real_values',
       'predicted_values']

pred_train = pred_train.reindex(columns=seq)

for i in range(0 , len(y_train)):
    real_values = y_train[i]
    predicted_values = p1[i]
    
    pred_train.iat[i,0] = real_values
    pred_train.iat[i,1] = predicted_values

pred_train = pred_train.dropna() #exclude any rows with missing data
pred_train.to_csv('~/data/predict_RForest_train.csv', index = True) 


# In[24]:


# Real values for the training set of x_test
print("Real values are:")
print(y_test)


# In[26]:


# Predicted values for the training set of x_test
p2 = rf.predict(x_test)
print("Predicted values are:")
print(p2)


# In[35]:


pred_test = pd.DataFrame({'real_values':[0.0]*len(y_test),
                     'predicted_values':[0.0]*len(y_test)})

#sort columns for convience
seq = ['real_values',
       'predicted_values']

pred_test = pred_test.reindex(columns=seq)

for i in range(0 , len(y_test)):
    real_values = y_test[i]
    predicted_values = p2[i]
    
    pred_test.iat[i,0] = real_values
    pred_test.iat[i,1] = predicted_values

pred_test = pred_test.dropna() #exclude any rows with missing data
pred_test.to_csv('~/data/predict_RForest_test.csv', index = True)


# In[27]:


# Calculating accuracy with score method for prediction of x_train
sc1 = rf.score(x_train, p1)
print(sc1)


# In[28]:


# Calculating accuracy with score method for prediction of x_test
sc2 = rf.score(x_test, p2)
print(sc2)


# In[30]:


# Making confusion matrix for x_train
from sklearn import metrics
predictions = rf.predict(x_train)
cm = metrics.confusion_matrix(y_train, predictions)
print(cm)


# In[31]:


# Making confusion matrix for x_test
from sklearn import metrics
predictions1 = rf.predict(x_test)
cm1 = metrics.confusion_matrix(y_test, predictions1)
print(cm1)


# In[32]:


# Random Forest regression worked much better than logistic regression model. It can be seen that the accuracy score in 
# random forest model is 1 for both training set and test set. That means they are 100% accurate. On the other hand,
# logistic regression model had an accuracy score 0.933.

