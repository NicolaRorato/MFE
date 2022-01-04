#!/usr/bin/env python
# coding: utf-8

# # Problem Set 7

# In[1]:


import os
# change path before running
os.chdir(r"C:\Users\Nicola\Documents\UNIVERSITY\Master\UCLA\MFE\Courses\Investments\Homework\HW7")
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.stats as ss
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
from functools import reduce
import quadprog as QD
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Prepare data
#load data 
#stock data
df_list = []
for i in range(0,5):
    df = pd.read_excel('lecture6p_2021.xlsx', sheet_name = i)
    #new_df = function(df)
    df_list.append(df)

#load Fama French data
ff = pd.read_excel('lecture6p_2021.xlsx', sheet_name = "F-F_Research_Data_Factors_daily")

begindate = "1989-12-29"
enddate = "2021-08-31"
def cleandt(d):
    d.columns = d.columns.str.lower()
    filt = (d['date'] >= begindate) & (d['date'] <= enddate)
    d = d.loc[filt]
    d = d[["date","adjclose"]]
    return d

df_list_new = [cleandt(d) for d in df_list]

df_list_new[0] = df_list_new[0].rename(columns = { "adjclose":"MSFT"})
df_list_new[1] = df_list_new[1].rename(columns = { "adjclose":"INTC"})
df_list_new[2] = df_list_new[2].rename(columns = { "adjclose":"LUV"})
df_list_new[3] = df_list_new[3].rename(columns = { "adjclose":"MCD"})
df_list_new[4] = df_list_new[4].rename(columns = { "adjclose":"JNJ"})

df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['date'], how='outer'), df_list_new)

# clean fama french data 
ff.columns = ff.columns.str.lower()
ff = ff[["date", "mktrf", "rf"]]
ff["rf"] = ff["rf"]/100
ff["mktrf"] = ff["mktrf"]/100
ff['rfleveladj'] = (ff['rf']+1).cumprod()
ff["mktrfleveladj"] = (ff['mktrf']+1).cumprod()
filt = (ff['date'] >= begindate) & (ff['date'] <= enddate)
ff = ff.loc[filt]

d = pd.merge(ff, df_merged, how="inner", on=["date"])

d['Date'] = d['date']
d = d.set_index('date')
d = d.resample('W-Fri').last()


d['rf'] = (d['rfleveladj']/d['rfleveladj'].shift() -1).shift()  #recreate the new rf
d['mktrf'] = (d['mktrfleveladj']/d['mktrfleveladj'].shift() -1)#recreate the excess market return
d['MSFT'] = (d['MSFT']/d['MSFT'].shift() -1)
d['INTC'] = (d['INTC']/d['INTC'].shift() -1)
d['LUV'] = (d['LUV']/d['LUV'].shift() -1)
d['MCD'] = (d['MCD']/d['MCD'].shift() -1)
d['JNJ'] = (d['JNJ']/d['JNJ'].shift() -1)

d= d.dropna()


# ## Question 1

# **Construct weekly simple total returns from the price data (use Adj Close to include dividends).  Compute and report the weekly and annualized mean and standard deviation for each stock. Compute the correlation matrix.**

# In[3]:


# Question 1 
dftable = d.iloc[:,4:9] #select all stocks 
def annret(x):
    result = (1+ x.mean())**52-1
    return result

def sd1y(x):
    result = x.std()*52**0.5
    return result

#standard deviation
res = dftable.agg([np.mean,np.std,annret, sd1y])
res*100


# In[4]:


# creation of correlation matrix
corrM = dftable .corr()
corrM*100


# ## Question 2

# **Construct the mean-variance frontier for the Intel-Microsoft combination. Indicate the minimum-variance portfolio and the efficient frontier (the efficient frontier is a set of expected returns - risks that you would want to consider investing in).**
# 
# Start by creating a some generic functions for computing the minimum variance frontier.

# In[5]:


#Start by creating a some generic functions for computing the minimum variance frontier.
def setupminvol(z,S):
    n = len(z)
    some1s = np.repeat(1,n).astype("double")
    some0s = np.repeat(0,n).astype("double")
    A = np.c_[some1s,z].astype("double")
    Si = np.linalg.inv(S)
    return {'S':S, 'Si':Si, 'z':z, 'n':n, 'some1s':some1s, 'some0s':some0s, 'some1s':some1s, 'A':A}

def minvol(targ, p):
    result =  QD.solve_qp(2*p['S'],p['some0s'], p['A'], np.array([1,targ]),meq=2)
    return result[1] #value 


def minvolw(targ, p):
    result = QD.solve_qp(2*p['S'],p['some0s'], p['A'], np.array([1,targ]),meq=2)
    return result[0] #solution - 2 values 


targstocks2 = dftable[["MSFT", "INTC"]]
S2 = targstocks2.cov().to_numpy() #variance covariance matrix for MSFT and INTC
z2 = targstocks2.agg([np.mean]).to_numpy().T

p2 = setupminvol(z2, S2)
delta = 10**(-7)
targrets = np.arange(0.002,0.0070001,delta)[0:50000]

def targfunc(targ):
    return minvol(targ,p2)

targfunc_vec = np.vectorize(targfunc)
vols2 = targfunc_vec(targrets)
vols2 = vols2**0.5

#identify the global min var portfolio
volmin2 = min(vols2) #find the global min variance portfolio
rmin2 = targrets[vols2 == volmin2]
pos2 = np.array(["2" if x <= rmin2 else "1" for x in targrets])
data = np.c_[vols2,targrets,pos2]
df2 = pd.DataFrame(data = data, columns = ['x', 'y','pos'])
df2["2stocks"] = "2stocks"

df2['x'] = df2['x'].astype('float')
df2['y'] = df2['y'].astype('float')
df2['pos'] = df2['pos'].astype('int')
#this will make our graph look cleaner
x = np.append(np.sqrt(np.diag(p2['S'])),volmin2)
y = np.append(p2['z'],rmin2)
pt = np.array(["MSFT","INTC","globalmin"])
data = np.c_[x,y,pt]
specialpoints2 = pd.DataFrame(data, columns = ['x', 'y','pt'])
specialpoints2['x'] = specialpoints2['x'].astype("float").round(5)
specialpoints2['y']  = specialpoints2['y'].astype("float").round(5)


# Next, run the functions for the two focal stocks and plot the results:

# In[6]:


##################################
# Plot Question 2
pos1 = df2.loc[(df2.pos == 1)]
pos2 = df2.loc[(df2.pos == 2)]

points = specialpoints2
cdict = {0: 'blue', 1: 'green', 2: 'red'}
for g, row in points.iterrows():
    plt.scatter(row.x, row.y,c = cdict[g])

plt.legend(points['pt'])
plt.plot(pos1.x,pos1.y, color = 'lightsteelblue')
plt.plot(pos2.x,pos2.y, linestyle = 'dashed', color = 'lightsteelblue')
plt.xlim(0.02, 0.07)
plt.ylim(0.002, 0.006)


# ## Question 3

# **Add remaining stocks to the mix. Compute the mean-variance frontier and plot it on the same chart with the one from the previous question. Indicate the minimum-variance portfolio and the efficient frontier. How do they compare to those of the previous question?**
# 
# We already did most of the workin in the previous question. Adding more assets is just a matter of calling the functions for more stocks. The additional stocks move the minimum frontier upward and inward. Of course, the marginal effect declines as the number of assets increases.

# In[9]:


targstocks5 = dftable
S5 = targstocks5.cov().to_numpy() #variance covariance matrix for MSFT and INTC
z5 = targstocks5.agg([np.mean]).to_numpy().T

p5 = setupminvol(z5, S5)
#delta = 10**(-7)
#targrets = np.arange(0.002,0.0070001,delta)[0:50000]

def targfunc(targ):
    return minvol(targ,p5)

targfunc_vec = np.vectorize(targfunc)
vols5 = targfunc_vec(targrets)
vols5 = vols5**0.5

#identify the global min var portfolio
volmin5 = min(vols5) #find the global min variance portfolio
rmin5 = targrets[vols5 == volmin5]
pos5 = np.array(["2" if x <= rmin2 else "1" for x in targrets])
data = np.c_[vols5,targrets,pos5]
df5 = pd.DataFrame(data = data, columns = ['x', 'y','pos'])
df5['x'] = df5['x'].astype('float')
df5['y'] = df5['y'].astype('float')
df5['pos'] = df5['pos'].astype('int')
df5["frontier"] = "5 stocks"
df2.rename(columns = {'2stocks':'frontier'}, inplace = True)
df25 = df5.append(df2, ignore_index = True )

#df2["2stocks"] = "2stocks"

#this will make our graph look cleaner
x = np.append(np.sqrt(np.diag(p5['S'])),[volmin2,volmin5])
y = np.append(p5['z'],[rmin2, rmin5])
pt = np.array(["MSFT","INTC","LUV","MCD","JNJ","2-stocks min","5-stocks min"])
data = np.c_[x,y,pt]
specialpoints2 = pd.DataFrame(data, columns = ['x', 'y','pt'])
specialpoints2['x'] = specialpoints2['x'].astype("float").round(5)
specialpoints2['y']  = specialpoints2['y'].astype("float").round(5)


# In[10]:


# Plot q3
pos1 = df25.loc[(df25.frontier == "2stocks")]
pos2 = df25.loc[(df25.frontier == "5 stocks")]

points = specialpoints2
#cdict = {0: 'blue', 1: 'green', 2: 'red'}
for g, row in points.iterrows():
    plt.scatter(row.x, row.y) #c = cdict[g]

plt.legend(points['pt'],loc = "upper left")
plt.xlim(0.02, 0.07)
plt.ylim(0.002, 0.007)


plt.plot(pos1.x,pos1.y, linestyle = 'dashed', color = 'lightsteelblue')
plt.plot(pos2.x,pos2.y, linestyle = 'solid', color = 'lightsteelblue')


# # Question 4

# **Add the riskless asset and construct the tangent portfolio for the Intel-Microsoft case. Next, construct the tangent portfolio for the full set of stocks. Compare the Sharpe ratios of the two tangent portfolios.**

# In[11]:


rfr = d['rf'].mean()
sharpe2 = (targrets-rfr)/vols2
sharpe5 = (targrets-rfr)/vols5
  
tang2 = np.where(sharpe2==max(sharpe2))
tang5 = np.where(sharpe5==max(sharpe5))

x= np.append(vols2[tang2[0]],vols5[tang5[0]])
y= np.append(targrets[tang2[0]],targrets[tang5[0]])
intercepts = np.append(rfr,rfr)
slopes = np.append(sharpe2[tang2[0]], sharpe5[tang5[0]])
portfolio = np.append("Sharpe ratio ="+str(sharpe2[tang2[0]]), "Sharpe ratio ="+str(sharpe5[tang5[0]]))

data = np.c_[x,y,intercepts,slopes,portfolio]
specialpoints4 = pd.DataFrame(data, columns = ['x', 'y','intercepts','slopes','portfolio'])
specialpoints4['x'] = specialpoints4['x'].astype("float").round(5)
specialpoints4['y']  = specialpoints4['y'].astype("float").round(5)
specialpoints4['intercepts']  = specialpoints4['intercepts'].astype("float").round(5)
specialpoints4['slopes']  = specialpoints4['slopes'].astype("float").round(5)

### Run this whole chunk to plot 
pos1 = df25.loc[(df25.frontier == "2stocks")]
pos2 = df25.loc[(df25.frontier == "5 stocks")]

#add 2 points of sharpe to special points2 
twopt= specialpoints4[['x','y','portfolio']]
twopt.rename(columns={'portfolio':'pt'}, inplace=True)
specialpoints2 = specialpoints2.append(twopt)


# In[12]:


#Plot Q4
points = specialpoints2
#cdict = {0: 'blue', 1: 'green', 2: 'red'}
for g, row in points.iterrows():
    plt.scatter(row.x, row.y) #c = cdict[g]

plt.legend(points['pt'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(0.02, 0.07)
plt.ylim(0.002, 0.007)

plt.plot(pos1.x,pos1.y, linestyle = 'dashed', color = 'lightsteelblue')
plt.plot(pos2.x,pos2.y, linestyle = 'solid', color = 'lightsteelblue')

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
    
abline(specialpoints4['slopes'].iloc[0], specialpoints4['intercepts'].iloc[0])
abline(specialpoints4['slopes'].iloc[1], specialpoints4['intercepts'].iloc[1])


# # Question 5

# **Assume your risk aversion is A = 4: What is your optimal mix of assets?**

# $$
# \begin{aligned}
#   w_{risky} &= \frac{E(R_{tang}) - R_{f}}{A \cdot V(R_{tang})}
# \end{aligned}
# $$
# 
# Compute the portfolio weight as the product of the tangency weights and the weight on the tangency portfolio. The risk-free portfolio weight is the residual. The negative weight implies borrowing, while a positive rate would have implied a risk-free bond investment.

# In[13]:


# Question 5 ####
#get the tangency weights
A = 4
w5 = minvolw(targrets[tang5[0]][0],p5)
#compute the weight on the risky and risk-free portfolio
wrisky = (targrets[tang5[0]][0] - rfr)/(A*vols5[tang5[0]][0]**2)
wrfr = 1 - wrisky
res3 = pd.DataFrame(np.zeros((6,2)), columns =["wtangent", "w (A = 4)"])
res3.index = ["msft", "intc", "luv", "mcd","jnj","RFR"]
res3.wtangent = np.append(w5,0)
res3["w (A = 4)"] = np.append(wrisky*w5,wrfr)
res3['wtangent'] = res3['wtangent'].astype("float")
round(res3*100,2)


# The weight on the risky portfolio is 125.3, implying a weight of -25.3 on the risk-free asset.

# In[ ]:




