import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datetime import date, timedelta

def daterange(start_date, end_date):
    dates = []
    for n in range(int((end_date - start_date).days)):
        dates.append(start_date + timedelta(n))

    return dates 

start = date(2025, 10, 1)
end = date(2026, 5, 1)

dates = daterange(start,end)

rng = np.random.default_rng(0)

par = np.array([0.15,0.1]) #beta,gamma

delta_t = 0.01
ts_cont = np.arange(0,len(dates),delta_t)

xs = np.zeros((len(ts_cont),3))
xs[0,:] = np.array([495.,5., 0.]) #Set the initial condition

def f(X,params):
    S,I,R = X
    N = S + I + R
    beta,gamma = params 
    dS = -beta * S * I/N
    dI = beta * S * I/N - gamma * I
    dR = gamma * I
    return np.array([dS,dI,dR])

for t_index in range(1,len(ts_cont)): 
    x_prev = xs[t_index - 1]
    xs[t_index] = x_prev + delta_t * f(x_prev,par)

plt.title('SIR Equations')
plt.plot(ts_cont,xs,label = ['S','I','R'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

data = rng.poisson(xs[::int(1/delta_t),1])

plt.title('Data')
plt.scatter(dates,data, marker = 'x')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

data = np.concatenate((np.array(dates)[...,np.newaxis],data[...,np.newaxis]),axis = -1)

pd.DataFrame(data,columns = ['date','count']).to_csv('fictious_data.csv',index = False)