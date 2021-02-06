import numpy as np


def Ornstein_Ulhenbeck(T, dt, mu, sigma, tau, Y0):
	# Initializations
	Y = list()
	t = np.arange(0, T, dt)
	Y.append(Y0)
	# Parameters
	N = len(t)
	sigma_term = sigma*np.sqrt(2.0/tau)*np.sqrt(dt)
	normal_draws = np.random.normal(loc=0.0, scale=1.0, size=N)
	# Integration
	for i in range(1, N):
		Y.append(Y[-1] - dt*(Y[-1]-mu)/tau + sigma_term*normal_draws[i])
	return np.array(Y)




def geometric_Ornstein_Ulhenbeck(T, dt, mu, sigma, tau, Y0):
	# Initializations
	Y = list()
	t = np.arange(0, T, dt)
	Y.append(Y0)
	# Parameters
	N = len(t)
	sigma_term = sigma*np.sqrt(dt)
	normal_draws = np.random.normal(loc=0.0, scale=1.0, size=N)
	# Integration
	for i in range(1, N):
		Y.append(Y[-1] - dt*(Y[-1]-mu)/tau + sigma_term*Y[-1]*normal_draws[i])
	return np.array(Y)








### Mean
T = 1000
dt = 1
mu = 0
sigma = 0.002
tau = 1

path_mean = Ornstein_Ulhenbeck(T=T, dt=dt, mu=mu, sigma=sigma, tau=tau, Y0=0)


import matplotlib.pyplot as plt

plt.plot(realization)
plt.show()



### Volatility
