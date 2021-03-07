#!/usr/bin/env python
# coding: utf-8

# ### Heuristic Models
# Look at the Seattle weather in the **data** folder. Come up with a heuristic model to predict if it will rain today. Keep in mind this is a time series, which means that you 
# only know what happened historically (before a given date). One example of a heuristic model is: It will rain tomorrow if it rained more than 1 inch (>1.0 PRCP) today. 
# Describe your heuristic model in the next cell.

# **my model here**  
# 
# Examples:  
# 
# If rained yesterday it will rain today.  
# If it rained yesterday or the day before it will rain today.

# In[3]:


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
                             'correct':[False]*numrows}) #TRUE if your guess matches the historical observation

#sort columns for convience
seq = ['yesterday','today','tomorrow','guess','rain_tomorrow','correct']
heuristic_df = heuristic_df.reindex(columns=seq)


# In[4]:


df.head()


# In[5]:


heuristic_df.head()


# Build a loop to add your heuristic model guesses as a column to this dataframe

# In[6]:


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
#    rain_tomorrow = df.iloc[i,1]
    rain_tomorrow = df.iloc[(i),1]
    
    heuristic_df.iat[z,0] = yesterday
    heuristic_df.iat[z,1] = today
    heuristic_df.iat[z,2] = tomorrow
    heuristic_df.iat[z,3] = False # set guess default to False
    heuristic_df.iat[z,4] = rain_tomorrow
    
#print(heuristic_df)
    
    
    ######### uncomment and create your heuristic guess ################
    #if ##### your conditions here #########:
    #    heuristic_df.iat[z,3] = True 
    ####################################################################
    
    # if it rains today or rained yesterday, it will rain tomorrow
    if heuristic_df.iat[z,1] > 0.0 or heuristic_df.iat[z,0] > 0.0:
        heuristic_df.iat[z,3] = True
    
    if heuristic_df.iat[z,3] == heuristic_df.iat[z,4]:
        heuristic_df.iat[z,5] = True
    else:
        heuristic_df.iat[z,5] = False


# In[7]:


heuristic_df.to_csv('~/data/heuristic_model.csv', index = True) 


# ### Evaluate the performance of the Heuristic model

# ***the accuracy of your predicitions***

# In[8]:


heuristic_df['correct'].value_counts()/numrows

