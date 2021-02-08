import yfinance as yf


sp500 = yf.Ticker("^GSPC")
data = sp500.history(period="10y", interval="1d")

import matplotlib.pyplot as plt




selected_data = data[data.index>="2015-01-01"]
selected_data = selected_data[selected_data.index<="2020-12-31"]




returns = selected_data["Close"].pct_change().to_numpy()[1:]
N = len(returns)


import numpy as np


n_profit = int(0.51*N)
n_loss = N-n_profit



aux = np.array([1]*n_profit + [-1]*n_loss)
hit = np.random.permutation(aux)


strategy_returns = np.abs(returns) * hit


# Portfolio growth
portfolio_growth = 1+np.cumsum(strategy_returns)


# Plot
#fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#axes[0].plot(selected_data["Close"], color=(0.8,0.5,0.5,1.0))
#axes[0].set_xlabel("Date")
#axes[0].set_ylabel("SP500")
#axes[1].plot(portfolio_growth, color=(0.5,0.5,0.8,1.0))
#axes[1].set_xlabel("Date")
#axes[1].set_ylabel("Portfolio value")
#plt.show()

# Metrics
total_return = np.sum(strategy_returns)
sharp_ratio = np.sqrt(252)*np.mean(strategy_returns)/np.std(strategy_returns)



print("Strategy total return =", np.round(total_return, 2))
print("Strategy Sharp ratio  =", np.round(sharp_ratio, 2))




##### Monte Carlo


def calculate_performance_realizations(returns, hit_ratio, backtest_length, n_realizations):
	realizations = list()
	total_returns = list()
	sharp_ratios = list()

	N = len(returns)

	n_profit = int(hit_ratio*N)
	n_loss = N-n_profit

	aux = np.array([1]*n_profit + [-1]*n_loss)

	for i in range(0, n_realizations):
		hit = np.random.permutation(aux)

		strategy_returns = np.abs(returns)*hit

		strategy_returns = np.random.permutation(strategy_returns)[0:backtest_length]

		realizations.append(strategy_returns)
		total_returns.append(np.sum(strategy_returns))
		sharp_ratios.append(np.sqrt(252)*np.mean(strategy_returns)/np.std(strategy_returns))

	return (np.array(realizations), np.array(total_returns), np.array(sharp_ratios))





(realizations, total_returns, sharp_ratios) = calculate_performance_realizations(returns=returns, hit_ratio=0.50, backtest_length=252, n_realizations=10000)


percentage_profitable = np.sum(total_returns>=0)/len(total_returns)
print(percentage_profitable)


## Sharp ratio statistics
pdf, bins, patches = plt.hist(x=sharp_ratios, bins=int(len(sharp_ratios)**(1/3)), density=True)

pdf = pdf / np.sum(pdf)
cmf = np.cumsum(pdf)



fig, axes = plt.subplots(1, 1, figsize=(5, 4))
axes.plot(bins[0:-1], 1-cmf, color=(0.8,0.5,0.5,1.0))
axes.set_xlabel("Sharp ratio")
axes.set_ylabel("1-Cumulative mass function")
#axes.set_yscale("log")
plt.show()


#### Case

case_index = np.argsort(sharp_ratios)[-3]
case_realization = realizations[case_index, :]
portfolio_growth = 1+np.cumsum(case_realization)


plt.plot(portfolio_growth)
plt.show()



dfgdf












###### Surface plot

hit_ratios = np.linspace(0.5, 0.6, 10)
backtest_lengths = np.arange(50, 252, 10)


ratio_profitable_strategies = list()

for hit_ratio in hit_ratios:
	print(hit_ratio)
	aux = list()
	for backtest_length in backtest_lengths:

		(realizations, total_returns, sharp_ratios) = calculate_performance_realizations(returns=returns, hit_ratio=hit_ratio, backtest_length=backtest_length, n_realizations=1000)

		percentage_profitable = np.sum(total_returns>=0)/len(total_returns)

		aux.append(percentage_profitable)

	ratio_profitable_strategies.append(aux)

ratio_profitable_strategies = np.array(ratio_profitable_strategies).T



### Plot

fig, axes = plt.subplots(1, 1, figsize=(5, 4))

# handle for displaying the colorbar
Z = [[0,0],[0,0]]
levels = np.linspace(hit_ratios[0], hit_ratios[-1], 40)


# color options
colormap_set = 'coolwarm'

from matplotlib import cm


colormap = cm.ScalarMappable(norm=None, cmap=colormap_set)
colormap.set_clim(vmin=np.min(hit_ratios), vmax=np.max(hit_ratios))
dummy = axes.contourf(Z, levels, cmap=colormap_set)
axes.cla()
cbar = plt.colorbar(dummy, format='%.2f')
cbar.set_label("Hit ratio", rotation=90)
##### data plotting
for i in range(0, len(hit_ratios)):
	rgb_code = colormap.to_rgba(x=hit_ratios[i], alpha=0.999, bytes=False, norm=True)
	axes.plot(backtest_lengths, ratio_profitable_strategies[:,i], color=rgb_code, linewidth=1.5, alpha=0.3)	


plt.show()
