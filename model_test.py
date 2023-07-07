#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle 
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\maqpr\Downloads\CAR DETAILS FROM CAR DEKHO.csv")


# In[3]:


df


# In[6]:


model = pickle.load(open(r"C:\Users\maqpr\Downloads\LinearRegressionModel.pkl",'rb'))


# In[7]:


#testing
model.predict(pd.DataFrame(columns=['name','year','km_driven','fuel','seller_type','transmission','owner'],data=np.array(['Renault KWID RXT',2016,40000,'Petrol','Individual','Manual','Firts Owner']).reshape(1,7)))


# In[8]:


model.predict(pd.DataFrame(columns=['name','year','km_driven','fuel','seller_type','transmission','owner'],data=np.array(['Hyundai i20 Magna 1.4 CRDi',2014,80000,'Diesel','Individual','Manual','Second Owner']).reshape(1,7)))



# In[ ]:




