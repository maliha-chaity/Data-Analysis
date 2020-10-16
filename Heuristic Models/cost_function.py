#!/usr/bin/env python
# coding: utf-8

# ### Heuristic Models (Cost Function Extension)
# Look at the Seattle weather in the **data** folder. Come up with a heuristic model to predict if it will rain today. Keep in mind this is a time series, which means that you only know what happened historically (before a given date). One example of a heuristic model is: It will rain tomorrow if it rained more than 1 inch (>1.0 PRCP) today. Describe your heuristic model in the next cell.

# **your model here**  
# 
# Examples:  
# 
# If rained yesterday it will rain today.  
# If it rained yesterday or the day before it will rain today.

# In[1]:


#here is an example of how to build and populate a hurestic model

import pandas as pd

df = pd.read_csv('~/data/seattle_weather_1948-2017.csv')

numrows = 25549 # can be as large as 25549

#create an empty dataframe to hold 100 values
heuristic_df = pd.DataFrame({'yesterday':[0.0]*numrows,
                             'today':[0.0]*numrows,
                             'tomorrow':[0.0]*numrows,
                             'guess':[False]*numrows, #logical guess
                             'rain_tomorrow':[False]*numrows, #historical observation
                             'correct':[False]*numrows, #TRUE if your guess matches the historical observation
                             'true_positive':[False]*numrows, #TRUE If you said it would rain and it did
                             'false_positive':[False]*numrows,#TRUE If you sait id would rain and it didn't
                             'true_negative':[False]*numrows, #TRUE if you said it wouldn't rain and it didn't
                             'false_negative':[False]*numrows}) #TRUE if you said it wouldn't raing and it did

#sort columns for convience
seq = ['yesterday',
       'today',
       'tomorrow',
       'guess',
       'rain_tomorrow',
       'correct',
       'true_positive',
       'false_positive',
       'true_negative',
       'false_negative']
heuristic_df = heuristic_df.reindex(columns=seq)


# In[2]:


df.head()


# In[3]:


heuristic_df.head()


# Build a loop to add your heuristic model guesses as a column to this dataframe

# In[4]:


# here is an example loop that populates the dataframe created earlier
# with the total percip from yesterday and today
# then the guess is set to true if rained both yesterday and today 
for z in range(numrows):
    #start at time 2 in the data frame
    i = z + 2
    #pull values from the dataframe
    yesterday = df.iloc[(i-2),1]
    today = df.iloc[(i-1),1]
    tomorrow = df.iloc[i,1]
    rain_tomorrow = df.iloc[(i),1]
    
    heuristic_df.iat[z,0] = yesterday
    heuristic_df.iat[z,1] = today
    heuristic_df.iat[z,2] = tomorrow
    heuristic_df.iat[z,3] = False # set guess default to False
    heuristic_df.iat[z,4] = rain_tomorrow
    
    #example hueristic
    if today > 0.0 and yesterday > 0.0:
        heuristic_df.iat[z,3] = True
        
    if heuristic_df.iat[z,3] == heuristic_df.iat[z,4]:
        heuristic_df.iat[z,5] = True
        if heuristic_df.iat[z,3] == True:
            heuristic_df.iat[z,6] = True #true positive
        else:
            heuristic_df.iat[z,8] = True #true negative
    else:
        heuristic_df.iat[z,5] = False
        if heuristic_df.iat[z,3] == True:
            heuristic_df.iat[z,7] = True #false positive
        else:
            heuristic_df.iat[z,9] = True #false negative


# ### Evaluate the performance of the Heuristic model

# In[5]:


print(heuristic_df)


# ***split data into training and testing***

# In[21]:


from sklearn.model_selection import train_test_split 

h_train, h_test = train_test_split(heuristic_df, test_size = 0.2, random_state = 0)
print("for training set:")
print(h_train.shape)
print("for test set:")
print(h_test.shape)


# ***the accuracy of your predicitions***

# In[22]:


# we used this simple approach in the first part to see what percent of the time we where correct 
# calculated as (true positive + true negative)/ number of guesses

print("accuracy for training set:")
print(h_train['correct'].value_counts()/(h_train.shape[0]))
print("accuracy for test set:")
print(h_test['correct'].value_counts()/(h_test.shape[0]))


# ***the precision of your predicitions***

# In[23]:


# precision is the percent of your postive prediction which are correct
# more specifically it is calculated (num true positive)/(num tru positive + num false positive)

print("precision for training set:")
print((h_train['true_positive'].value_counts()[0])/
(h_train['true_positive'].value_counts()[0] + h_train['false_positive'].value_counts()[0]))

print("precision for test set:")
print((h_test['true_positive'].value_counts()[0])/
(h_test['true_positive'].value_counts()[0] + h_test['false_positive'].value_counts()[0]))


# ***the recall of your predicitions***

# In[24]:


# recall the percent of the time you are correct when you predict positive
# more specifically it is calculated (num true positive)/(num tru positive + num false negative)

print("recall for training set:")
print((h_train['true_positive'].value_counts()[0])/
(h_train['true_positive'].value_counts()[0] + h_train['false_negative'].value_counts()[0]))

print("recall for test set:")
print((h_test['true_positive'].value_counts()[0])/
(h_test['true_positive'].value_counts()[0] + h_test['false_negative'].value_counts()[0]))


# ***The sum of squared error (SSE) of your predictions***

# In[27]:


import warnings
warnings.filterwarnings('ignore')

print("SSE for training set:")
h_train['square_error'] = (h_train['guess'].astype(int) - h_train['rain_tomorrow'].astype(int)) ** 2
print(h_train['square_error'].head(20))
print(h_train['square_error'].sum())

print("SSE for test set:")
h_test['square_error'] = (h_test['guess'].astype(int) - h_test['rain_tomorrow'].astype(int)) ** 2
print(h_test['square_error'].head(20))
print(h_test['square_error'].sum())

