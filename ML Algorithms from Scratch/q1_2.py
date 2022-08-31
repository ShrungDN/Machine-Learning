import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('A2Q1.csv', header=None)
data = df.to_numpy()
data = data[:,0]
n = len(data)
K = 4

plt.figure()
plt.plot(data, np.zeros_like(data), 'x')
plt.xlabel('Values of Data points')
plt.ylabel('None')
plt.title('Plot of dataset on a single axis')

plt.figure()
plt.hist(data, bins=200)
plt.xlabel('Values of Data points')
plt.ylabel('Number of data points in a range')
plt.title('Histogram of Dataset')

def gaussian_em(data, K):
  n = len(data)

  # rand initialization
  mu = np.random.uniform(low=-5, high=5, size = K)
  sigma2 = np.random.uniform(low=0, high=10, size = K)
  pi = np.random.uniform(low=0, high=1, size = K)
  pi = pi/np.sum(pi)
  pi[-1] = 1 - np.sum(pi[:-1])
  lam = np.zeros((K,n))   # lambda[k,i], assigning to zeros only to capture shape of matrix, it will be changed in next step

  iter = 0
  error_norm = 1
  log_likelihood_list = []
  while iter <=1000 and error_norm >= 0.01:
  # while iter <=1000 and error_norm >= 0.1:
    iter += 1
    # if iter%100 == 0:
      # print(iter)

    log_likelihood = 0
    epsilon1 = 10**-50
    for i in range(n):
      temp = 0
      for k in range(K):
        temp += pi[k] * (1/(np.sqrt(2*np.pi * sigma2[k]))) * (np.exp((-(data[i]-mu[k])**2)/(2*sigma2[k])))
      # print(iter, temp)
      log_likelihood += np.log(epsilon1 + temp)
      # log_likelihood += temp
    log_likelihood_list.append(log_likelihood)

    # fix theta, max lambda
    epsilon2 = 10**-100 # taking some epsilon and summing with exponential in next step, as in few cases the exponential becomes 0
    # like in the case of e^(-42)
    new_lam = lam.copy()
    for k in range(K):
      for i in range(n):
        new_lam[k,i] = epsilon2 + (1/(np.sqrt(2*np.pi * sigma2[k])) * (np.exp((-(data[i]-mu[k])**2)/(2*sigma2[k]))) * pi[k])
    for i in range(n):
      new_lam[:,i] = new_lam[:,i]/(np.sum(new_lam[:,i]))

    # fix lambda, max theta
    new_mu = np.array([np.sum(new_lam[k,:]*data)/np.sum(new_lam[k,:]) for k in range(K)])
    new_sigma2 = np.array([np.sum(new_lam[k,:]*((data-new_mu[k])**2))/np.sum(new_lam[k,:]) for k in range(K)])
    new_pi = np.array([np.sum(new_lam[k,:])/n for k in range(K)])

    error_norm = (np.linalg.norm(new_mu-mu))**2 + (np.linalg.norm(new_sigma2-sigma2))**2 + (np.linalg.norm(new_pi-pi))**2 

    mu = new_mu.copy()
    sigma2 = new_sigma2.copy()
    pi = new_pi.copy()
    lam = new_lam.copy()

    # print(iter, error_norm)

  # print(log_likelihood_list)
  return mu, sigma2, pi, lam, log_likelihood_list

log_likelihood_lists = []
min_iter = float('inf')
for i in range(100):
  mu, sigma2, pi, lam, log_likelihood = gaussian_em(data, K=4)
  log_likelihood_lists.append(log_likelihood)
  iter = len(log_likelihood)
  min_iter = min(iter, min_iter)

average_log_likelihood = np.zeros(min_iter)
for j in range(min_iter):
  for i in range(100):
    average_log_likelihood[j] += log_likelihood_lists[i][j]
  average_log_likelihood[j] /= 100

plt.figure()
# excluding first point as it is a random initialization
plt.plot(average_log_likelihood[1:], 'o') 
plt.xlabel('Iterations')
plt.ylabel('Average Log-likelihood')
plt.title('Average Log-likelihood over 100 random initializations')

def gaussian_distribution(x, mu, sigma):
  return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp((-(x-mu)**2)/(2*sigma**2))

plt.figure()
x = np.linspace(-np.min(data), np.max(data), 1000)
for k in range(K):
  plt.plot(x, gaussian_distribution(x,mu[k],np.sqrt(sigma2[k])))
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Distribution of Guassian Mixtures')

plt.show()