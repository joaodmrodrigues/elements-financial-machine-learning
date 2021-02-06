import numpy as np





def normalize(sample):
	return (sample-np.mean(sample))/np.std(sample)


def generate_samples(N, dependence_coef, which_param):
	### Cause
	samples_cause = np.random.normal(loc=0.0, scale=1.0, size=N)

	### Effect
	inherent_noise = np.random.normal(loc=0.5, scale=1.0, size=N)
	param = normalize((1-dependence_coef)*inherent_noise + dependence_coef*samples_cause)

	samples_effect = list()
	for i in range(0, N):
		if which_param == 1:
			mean = param[i]
			#scale = np.exp(np.random.normal(loc=np.random.normal(loc=0, scale=1.0)))
			scale = 1.0
		if which_param == 2:
			mean = np.random.normal(loc=np.random.normal(loc=0, scale=1.0))
			#mean = 0
			scale = np.exp(param[i])
	
		[samples_effect.append(np.random.normal(loc=mean, scale=scale))]

	return [samples_cause, samples_effect]









N = 10000
dependence_coef = 0.7



### Dependence on the mean
samples = generate_samples(N=N, dependence_coef=dependence_coef, which_param=1)
samples_cause1 = samples[0]
samples_effect1 = samples[1]

### Dependence on the scale
samples = generate_samples(N=N, dependence_coef=dependence_coef, which_param=2)
samples_cause2 = samples[0]
samples_effect2 = samples[1]


import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(samples_cause1, samples_effect1, '.', color=(0.8,0.5,0.5,0.2))
axes[0].set_xlabel("Cause")
axes[0].set_ylabel("Effect")
axes[0].set_title("Relation on the mean")
axes[1].plot(samples_cause2, samples_effect2, '.', color=(0.8,0.5,0.5,0.2))
axes[1].set_xlabel("Cause")
axes[1].set_ylabel("Effect")
axes[1].set_title("Relation on the scale")
plt.tight_layout()
plt.show()









def calculate_entropy(X):
	# 1) Histograms the samples
	nbins = int(len(X)**(1/3))
	p = np.histogram(X, bins=nbins, density=False)[0]
	p = p/np.sum(p)+1e-6
	# 2) Calculates the entropy
	entropy = -np.sum(p*np.log2(p))
	
	return entropy


def calculate_joint_entropy(X, Y):
	# 1) Histograms the samples
	nbins = int(len(X)**(1/3))
	p = np.histogram2d(X, Y, bins=nbins, density=False)[0]
	p = p/np.sum(p)+1e-6
	# 2) Calculates the entropy
	entropy = -np.sum(p*np.log2(p))
	
	return entropy

def calculate_mutual_information(X, Y):
	S_X = calculate_entropy(X)
	S_Y = calculate_entropy(Y)
	S_XY = calculate_joint_entropy(X, Y)
	I = S_X+S_Y-S_XY
	return I



mutual_information = calculate_mutual_information(X=samples_cause2, Y=samples_effect2)



















##########################################
##########################################



def calculate_mutual_information_score(X, Y, n_perm):
	# Mutual information on original samples
	I = calculate_mutual_information(X=X, Y=Y)

	# Mutual information on randomly shuffled data
	I_random = list()
	ind = np.arange(len(Y))
	for i in range(0, n_perm):
		np.random.shuffle(ind)
		Y_shuffled = Y[ind]
		I_random.append(calculate_mutual_information(X=X, Y=Y_shuffled))

	# Calculates the mutual information score
	mi_score = (I-np.mean(I_random))/np.std(I_random)

	return mi_score









n_perm = 100
#mi_score = calculate_mutual_information_score(X=samples_cause1, Y=samples_effect1, n_perm=n_perm)
#print(mi_score)




#########################################
#########################################

dependence_coefs = np.linspace(0, 1, 15)

mi_scores_mean = list()
mi_scores_scale = list()

for coef in dependence_coefs:

	### On mean
	samples = generate_samples(N=N, dependence_coef=coef, which_param=1)
	samples_cause = normalize(samples[0])
	samples_effect = normalize(samples[1])
	mi_scores_mean.append(calculate_mutual_information_score(X=samples_cause, Y=samples_effect, n_perm=n_perm))
	
	### On scale
	samples = generate_samples(N=N, dependence_coef=coef, which_param=2)
	samples_cause = normalize(samples[0])
	samples_effect = normalize(samples[1])

	mi_scores_scale.append(calculate_mutual_information_score(X=samples_cause, Y=samples_effect, n_perm=n_perm))




fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(dependence_coefs, mi_scores_mean, '-', color=(0.8,0.5,0.5,1.0))
axes[0].set_xlabel("Dependence coeficient")
axes[0].set_ylabel("Mutual information score")
axes[0].set_title("Relation on the mean")
axes[1].plot(dependence_coefs, mi_scores_scale, '-', color=(0.8,0.5,0.5,1.0))
axes[1].set_xlabel("Dependence coeficient")
axes[1].set_ylabel("Mutual information score")
axes[1].set_title("Relation on the scale")
plt.show()