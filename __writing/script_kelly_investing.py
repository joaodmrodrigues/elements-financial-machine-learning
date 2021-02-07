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

path_mean = Ornstein_Ulhenbeck(T=T, dt=dt, mu=mu, sigma=sigma, tau=tau, Y0=mu)


import matplotlib.pyplot as plt

#plt.plot(path_mean)
#xplt.show()



### Volatility
T = 1000
dt = 1
mu = 0.02
sigma = 0.01
tau = 10

path_std = geometric_Ornstein_Ulhenbeck(T=T, dt=dt, mu=mu, sigma=sigma, tau=tau, Y0=mu)

#plt.plot(path_std)
#plt.show()



### Price path
returns = np.random.normal(loc=path_mean, scale=path_std)
path_price = 100*np.cumprod(1+returns)


#plt.plot(path_price)
#plt.show()



### Calculates the hit ratio
hit_ratio = np.sum(np.sign(returns*path_mean)>0)/len(returns)
print(hit_ratio)






###################### Trading simulation
portfolio_initial = 1
leverage = 10


def calculate_portfolio_growth(portfolio_initial, fractions, price_returns):
	N = len(fractions)
	strategy_returns = fractions*price_returns
	portfolio_growth = portfolio_initial*np.cumprod(1+strategy_returns)

	if True in [value<=1e-2 for value in portfolio_growth]:
		bust_ind = np.min(np.where(portfolio_growth<1e-2)[0])
		portfolio_growth[bust_ind:] = 0

	return portfolio_growth



##### Strategy 1 - full leverage
fractions1 = leverage * np.sign(path_mean)
portfolio1 = calculate_portfolio_growth(portfolio_initial=portfolio_initial, fractions=fractions1, price_returns=returns)
##### Strategy 2 - half leverage
fractions2 = 0.5*leverage * np.sign(path_mean)
portfolio2 = calculate_portfolio_growth(portfolio_initial=portfolio_initial, fractions=fractions2, price_returns=returns)
##### Strategy 3 - Kelly
fractions3 = path_mean/path_std**2
fractions3[fractions3>leverage] = leverage
fractions3[fractions3<-leverage] = -leverage
portfolio3 = calculate_portfolio_growth(portfolio_initial=portfolio_initial, fractions=fractions3, price_returns=returns)




fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(path_price, '-', color=(0.8,0.5,0.5,1.0))
axes[0].set_xlabel("Day")
axes[0].set_ylabel("Price")
axes[1].plot(portfolio1, '-', label="Full leverage", color=(0.8,0.5,0.5,1.0))
axes[1].plot(portfolio2, '-', label="Half leverage", color=(0.5,0.5,0.8,1.0))
axes[1].plot(portfolio3, '-', label="Kelly leverage", color=(0.5,0.7,0.7,1.0))
#axes[1].legend()
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Portfolio value")
axes[1].set_yscale("log")
plt.show()










##### Sharp ratio
def calculate_sharp_ratio(portfolio):
	N = len(portfolio)

	if True in [value==0 for value in portfolio]:
		bust_ind = np.min(np.where(portfolio==0)[0])
		portfolio = portfolio[:bust_ind]

	returns = (portfolio[1:]-portfolio[0:-1])/portfolio[0:-1]
	sharp_ratio = np.sqrt(252)*np.mean(returns)/np.std(returns)
	return sharp_ratio



print("Mean Kelly fraction")
print(np.mean(np.abs(fractions3)))

print(calculate_sharp_ratio(portfolio1))
print(calculate_sharp_ratio(portfolio2))
print(calculate_sharp_ratio(portfolio3))





########## Ensemble
n_runs = 100

strategy1 = list()
strategy2 = list()
strategy3 = list()
mean_kelly_fraction = list()
hit_ratio = list()

for i in range(0, n_runs):

	# Price path
	path_mean = Ornstein_Ulhenbeck(T=1000, dt=1, mu=0, sigma=0.002, tau=1, Y0=0)
	path_std = geometric_Ornstein_Ulhenbeck(T=1000, dt=1, mu=0.02, sigma=0.01, tau=10, Y0=0.02)
	returns = np.random.normal(loc=path_mean, scale=path_std)
	path_price = 100*np.cumprod(1+returns)	


	# Full leverage
	fractions1 = leverage * np.sign(path_mean)
	portfolio1 = calculate_portfolio_growth(portfolio_initial=portfolio_initial, fractions=fractions1, price_returns=returns)
	strategy1.append(calculate_sharp_ratio(portfolio1))


	# Half leverage
	fractions2 = 0.5 * leverage * np.sign(path_mean)
	portfolio2 = calculate_portfolio_growth(portfolio_initial=portfolio_initial, fractions=fractions2, price_returns=returns)
	strategy2.append(calculate_sharp_ratio(portfolio2))


	# Kelly
	fractions3 = path_mean/path_std**2
	fractions3[fractions3>leverage] = leverage
	fractions3[fractions3<-leverage] = -leverage
	portfolio3 = calculate_portfolio_growth(portfolio_initial=portfolio_initial, fractions=fractions3, price_returns=returns)
	strategy3.append(calculate_sharp_ratio(portfolio3))

	# mean kelly fraction
	mean_kelly_fraction.append(np.mean(np.abs(fractions3)))

	# hit ratio
	hit_ratio.append(np.sum(np.sign(returns*path_mean)>0)/len(returns))



sharp_ratio = np.array([strategy1, strategy2, strategy3]).T
strategy_labels = ["Full leverage", "Half leverage", "Kelly optimal"]




fig, axes = plt.subplots(1, 1, figsize=(5, 4))
axes.boxplot(sharp_ratio)
axes.set_xticklabels(strategy_labels, rotation=90)
#axes.set_xlabel("Strategy")
axes.set_ylabel("Sharp ratio")
plt.tight_layout()
plt.show()