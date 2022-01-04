#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import datetime


# load data
df = pd.read_csv("sp500_2021.csv",sep = ',',parse_dates=['caldt'])

# average daily returns
df.set_index('caldt', inplace=True)
avg_daily=df['vwretd'].mean()
print("annualized arithmetic daily return is", avg_daily*252)

# average monthly returns
df_monthly=df.resample('MS').agg(lambda x: (x + 1).prod() - 1)
avg_monthly=df_monthly['vwretd'].mean()
print("annualized arithmetic monthly return is", avg_monthly*12)

# average yearly returns
df_yearly=df.resample('YS').agg(lambda x: (x + 1).prod() - 1)
avg_yearly=df_yearly['vwretd'].mean()
print("annualized arithmetic yearly return is", avg_yearly)

# arithmetic 5 yearly returns

df_new=df.reset_index()
new= (df_new['caldt'] > '1972-01-03') & (df_new['caldt'] <= '2016-12-30')
filterdf=df_new.loc[new]
filterdf.set_index('caldt', inplace=True)
filterdf_5year=filterdf.resample('5YS').agg(lambda x: (x + 1).prod() - 1)
avg_5yearly=filterdf_5year['vwretd'].mean()
print("annualized arithmetic 5 yearly return is", avg_5yearly/5)

# geometric daily returns

from statistics import geometric_mean
df_copy = df.copy()
df_copy['vwretd']=df_copy['vwretd']+1
mean=geometric_mean(df_copy['vwretd'])
geometric_daily=(mean)**252-1
print("annualized geometric daily return is", geometric_daily)

# geometric monthly returns

df_monthly['vwretd']=df_monthly['vwretd']+1
mean=geometric_mean(df_monthly['vwretd'])
geometric_monthly=(mean)**12-1
print("annualized geometric monthly return is", geometric_monthly)

# geometric yearly returns

df_yearly['vwretd']=df_yearly['vwretd']+1
mean=geometric_mean(df_yearly['vwretd'])
geometric_yearly=(mean-1)
print("annualized geometric yearly return is", geometric_yearly)

# geometric 5 yearly returns

filterdf5_copy = filterdf_5year.copy()
filterdf5_copy['vwretd']=filterdf5_copy['vwretd']+1
mean=geometric_mean(filterdf5_copy['vwretd'])
geometric_5year=(mean-1)
print("annualized geometric 5 yearly return is", geometric_5year/5)


# In[2]:


df_monthly


# In[5]:


bond = pd.read_csv("gsw_yields_2021.csv",sep = ',',parse_dates=['Date'])
bond_final=bond[['Date','BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']]
date= (bond_final['Date'] >= '1972-01-03') & (bond_final['Date'] <= '2020-12-31')
bond_final=bond_final.loc[date]
bond_set=bond_final.reset_index(drop=True)


# In[6]:


def nssyld(b0, b1, b2, b3, t1, t2, m):
    mt1 = m/t1
    mt2 = m/t2    
    expmt1 = np.exp(-mt1)
    expmt2 = np.exp(-mt2)
    c1 = (1-expmt1)/mt1
    c2 = c1 - expmt1
    c3 = (1-expmt2)/mt2 - expmt2
    return(np.exp((b0 + b1 * c1 + b2 * c2 + b3 * c3)/100)-1)


# In[7]:


bond_new= bond_set.dropna( how='any', subset=["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"])
bond_new.set_index('Date', inplace=True)


# In[8]:


# arithmetic daily returns

bond_daily=bond_new.copy()
bond_daily['yield']=nssyld(bond_daily['BETA0'],bond_daily['BETA1'],bond_daily['BETA2'], 
                           bond_daily['BETA3'], bond_daily['TAU1'], bond_daily['TAU2'], (1/365))
bond_daily

avgbond_daily=bond_daily['yield'].mean()
print("annualized arithmetic daily return is", avgbond_daily)

#  arithmetic monthly returns
bond_monthly=bond_new.copy()
bond_monthly['yield']=nssyld(bond_monthly['BETA0'],bond_monthly['BETA1'],bond_monthly['BETA2'], 
                             bond_monthly['BETA3'], bond_monthly['TAU1'], bond_monthly['TAU2'], (1/12))
bond_monthlyfinal=bond_monthly.resample('MS').first()
avg_bond_monthly=bond_monthlyfinal['yield'].mean()
print("annualized arithmetic monthly return is", avg_bond_monthly)

#  arithmetic yearly returns
bond_yearly=bond_new.copy()
bond_yearly['yield']=nssyld(bond_yearly['BETA0'],bond_yearly['BETA1'],bond_yearly['BETA2'], bond_yearly['BETA3'], 
                            bond_yearly['TAU1'], bond_yearly['TAU2'],1)
bond_yearlyfinal=bond_yearly.resample('YS').first()
avg_bond_yearly=bond_yearlyfinal['yield'].mean()
print("annualized arithmetic yearly return is", avg_bond_yearly)

#  arithmetic 5 yearly returns

bond_5yearly=bond_new.copy()
bond_5yearly['yield']=nssyld(bond_5yearly['BETA0'],bond_5yearly['BETA1'],bond_5yearly['BETA2'], 
                             bond_5yearly['BETA3'], bond_5yearly['TAU1'], bond_5yearly['TAU2'],5)
bond_yearly5final=bond_yearly.resample('5YS').first()
avg_bond_5yearly=bond_yearly5final['yield'].mean()
print("annualized arithmetic 5 yearly return is", avg_bond_5yearly)


# In[10]:


# geometric daily returns

from statistics import geometric_mean
bond_daily_copy = bond_daily.copy()
bond_daily_copy['yield']=bond_daily_copy['yield']+1
mean=geometric_mean(bond_daily_copy['yield'])
geometric_daily_bond=(mean)-1
print("annualized geometric daily return is", geometric_daily_bond)

# geometric monthly returns

bond_monthly_copy = bond_monthlyfinal.copy()
bond_monthly_copy['yield']=bond_monthly_copy['yield']+1
mean=geometric_mean(bond_monthly_copy['yield'])
geometric_monthly_bond=(mean)-1
print("annualized geometric monthly return is", geometric_monthly_bond)

# geometric yearly returns

bond_yearly_copy = bond_yearlyfinal.copy()
bond_yearly_copy['yield']=bond_yearly_copy['yield']+1
mean=geometric_mean(bond_yearly_copy['yield'])
geometric_yearly_bond=(mean)-1
print("annualized geometric yearly return is", geometric_yearly_bond)


# geometric 5 yearly returns

bond_yearly5final_copy = bond_yearly5final.copy()
bond_yearly5final_copy['yield']=bond_yearly5final_copy['yield']+1
mean=geometric_mean(bond_yearly5final_copy['yield'])
geometric_5yearly_bond=(mean)-1
print("annualized geometric 5 yearly return is", geometric_5yearly_bond)


# In[11]:


bondstock = {'Frequency':['daily','monthly','yearly','fiveyearly'],'Arithmetic Stock':[avg_daily*252,avg_monthly*12,avg_yearly,avg_5yearly/5],
             'Geometric Stock':[geometric_daily,geometric_monthly,geometric_yearly,geometric_5year/5],
             'Arithmetic Bond':[avgbond_daily,avg_bond_monthly,avg_bond_yearly,avg_bond_5yearly],
             'Geometric Bond':[geometric_daily_bond,geometric_monthly_bond,geometric_yearly_bond,geometric_5yearly_bond]}
result=pd.DataFrame(bondstock )


# In[12]:


# excess returns
result['excess returns arithmetic in %']=(result['Arithmetic Stock']-result['Arithmetic Bond'])*100
result['excess returns geometric in %']=(result['Geometric Stock']-result['Geometric Bond'])*100

result


# In[13]:


# average daily std deviation
avg_std_daily=df['vwretd'].std()
annualized_daily_std=avg_std_daily*(252)**0.5
print("annualized arithmetic daily std deviation is",annualized_daily_std*100,"%")


# In[14]:


wt=(result['excess returns arithmetic in %'][0]/100)/(4*(annualized_daily_std)**2)
wt


# In[15]:


print("This implies investor should allocate",round(wt,2)," to stocks and",round(1-round(wt,2),2)," to bonds, given A = 4.") 


# In[ ]:




