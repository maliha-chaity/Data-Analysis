#!/usr/bin/env python
# coding: utf-8

# ### In each cell complete the task using pandas

# In[ ]:


import pandas as pd
import numpy as np


# Read in the titanic.csv file in the `~/data` directory as a pandas dataframe called **df**

# In[2]:


df = pd.read_csv('~/data/titanic.csv')


# Display the head of the dataframe

# In[10]:


df.head()


# What is the percentage of people who survived? (hint find the mean of the survival column)

# In[25]:


total = df['Survived'].sum() 
print("total is:", total)

ind = df.index
rw = len(ind)
print("rw is:", rw)

per = (total*100) / rw
print(per)


# How many women and how many men survived?

# In[27]:


ml = df[(df["Sex"] == "male") & (df["Survived"] == 1)]
ml_sur = ml["Survived"].sum()
print("men survived:", ml_sur)


# In[28]:


fml = df[(df["Sex"] == "female") & (df["Survived"] == 1)]
fml_sur = fml["Survived"].sum()
print("women survived:", fml_sur)


# What is the percentage of people that survied who paid a fare less than 10?

# In[36]:


fr = df[(df["Fare"] <10) & (df["Survived"] == 1)]
fr.head()

rows = fr.index
rws = len(rows)

pr = (rws*100) / rw
print(pr)


# What is the average age of those who didn't survive?

# In[38]:


ag = df[(df["Survived"] == 0)] 
ag.head()
tot_age = ag["Age"].sum()

rw_ag = ag.index
rwa = len(rw_ag)

avg = tot_age / rwa
print(avg)


# What is the average age of those who did survive?

# In[40]:


ags = df[(df["Survived"] == 1)] 
ags.head()

ta = ags["Age"].sum()
rw_ags = ags.index
rwas = len(rw_ags)
av = ta / rwas
print(av)


# What is the average age of those who did and didn't survive grouped by gender?

# In[44]:


ms = df[(df["Survived"] == 1) & (df["Sex"] == "male")] 
ms.head()

ams = ms["Age"].sum()
rw_ms = ms.index
rms = len(rw_ms)
avms = ams / rms
print("Average age of male who survived:", avms)

##########################################
mns = df[(df["Survived"] == 0) & (df["Sex"] == "male")] 
mns.head()

amns = mns["Age"].sum()
rw_mns = mns.index
rmns = len(rw_mns)
avmns = amns / rmns
print("Average age of male who did not survive:", avmns)

##########################################
fs = df[(df["Survived"] == 1) & (df["Sex"] == "female")] 
fs.head()

afs = fs["Age"].sum()
rw_fs = fs.index
rfs = len(rw_fs)
avfs = afs / rfs
print("Average age of female who survived:", avfs)

##########################################
fns = df[(df["Survived"] == 0) & (df["Sex"] == "female")] 
fns.head()

afns = fns["Age"].sum()
rw_fns = fns.index
rfns = len(rw_fns)
avfns = afns / rfns
print("Average age of female who did not survive:", avfns)




# ## Tidy GDP

# Manipulate the GDP.csv file and make it tidy, the result should be a pandas dataframe with the following columns:
# * Country Name
# * Country Code
# * Year
# * GDP

# In[2]:


data = pd.read_csv('~/data/GDP.csv', skiprows = 4) 

data.drop(data.columns[[2,3]], axis = 1, inplace = True)

df1 = pd.melt(data, id_vars = ["Country Name", "Country Code"], var_name = "Year", value_name = "GDP")

df1.to_csv('~/data/modifiedGDP.csv', index = True) 







































































# In[ ]:




