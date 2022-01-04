#!/usr/bin/env python
# coding: utf-8

# The stock dynamics under the risk neutral probability $Q$ is
# $$\frac{dS_t}{S_t} = r dt + \sigma dW_t^Q$$

# In[4]:


import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def blackscholes(S0, sigma, T):
    d1 = sigma/2 * np.sqrt(T)
    Nd1 = scipy.stats.norm(0, 1).cdf(d1)
    #print(d1, Nd1)
    return S0*(2*Nd1 - 1)

S0 = 100
sigma = 0.2
mu = 0.05
r = 0
dt = 0.01
T = 1

N = 1000 # number of paths
t = int(T/dt) # number of intervals

# y=logS is normal with mean=logS0 + (r-sigma**2)(T-t)/2 and var=sigma**2(T-t)
S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt
+ sigma * np.sqrt(dt)
* np.random.standard_normal((t, N)), axis=0)) 

plt.figure()
plt.plot(S[:,:10])


# In[5]:


# a
K = 100

C = sum(np.maximum(S[-1] - K, 0)) / N
C_bs = blackscholes(S0, sigma, T)
print('Simulated option price', C)
print('Analytical option price', C_bs)

plt.figure()
plt.hist(S[-1])


# In[6]:


# b
avg_S = S.mean(axis=0)
C = sum(np.maximum(avg_S - K, 0)) / N
if C == 0:
    ST = S.mean()
    print(f'The Average Stock Price at maturity {ST} is smaller than 100')
else:
    pass

print('Simulated Asian option price', C)

plt.figure()
plt.hist(avg_S)

