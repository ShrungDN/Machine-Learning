import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the csv file
df = pd.read_csv('Dataset.csv', header=None)
data = df.to_numpy()
data = data.transpose()

# Kernel Functions
def kernelA(x, y, d):
  return (1+np.dot(x, y))**d

def kernelB(x, y, sigma):
  return np.exp(-(np.dot((x-y),(x-y)))/(2*(sigma**2)))

# Class to perform Spectral Clustering (Spectral Relaxation of K-means)
class Spectral_Clustering_1:
  def __init__(self, data, num_clusters, kernel, kernel_param):
    # X0 is the data
    # N is the number of data points
    # kernel is the kernel function that is input
    # kernel_param is a parameter that the kernel function uses
    # k the number of clusters to be formed
    self.X0 = data.copy()
    Xtemp = data.copy()
    self.kernel = kernel
    self.param = kernel_param
    self.N = len(Xtemp[0])
    self.k = num_clusters

    # X and Y axis limits of plot
    self.xlim = np.array([np.min(self.X0[0])-0.1*np.absolute(np.min(self.X0[0])), np.max(self.X0[0])+0.1*np.absolute(np.max(self.X0[0]))])
    self.ylim = np.array([np.min(self.X0[1])-0.1*np.absolute(np.min(self.X0[1])), np.max(self.X0[1])+0.1*np.absolute(np.max(self.X0[1]))])

    # Centering the data
    Xs = Xtemp.sum(axis=1)
    Xs = np.expand_dims(Xs, axis=0)
    Xs = Xs.transpose()
    Xtemp = Xtemp - Xs/self.N

    # Calculating Kernel Matrix
    K = np.zeros((self.N, self.N))
    for i in range(self.N):
      for j in range(self.N):
        K[i,j] = self.kernel(Xtemp[:,i], Xtemp[:,j], kernel_param)  

    # Centering Kernel Matrix (Centering data in higher dimension)
    cmat = np.ones((self.N, self.N))/self.N #centering matrix
    K = K - np.matmul(cmat, K) - np.matmul(K, cmat) + np.matmul(np.matmul(cmat,K), cmat) 

    # Calculating top eigen vectors to be used for spectral relaxation of K-means
    eigval, eigvec = np.linalg.eig(K)
    eigvec = eigvec.transpose()
    # Ignoring imaginary part as it is negligible
    eigvec = eigvec.real
    self.eig = list(zip(eigval, eigvec))
    self.eig.sort(reverse=True)
    self.topeig = self.eig[:self.k]

    # Creating a new data set for running K-means algorithm on
    self.X = np.array([val_vec[1] for val_vec in self.topeig])
    for i in range(self.N):
      self.X[:,i] = self.X[:,i]/np.linalg.norm(self.X[:,i])

    # Running K-means algorithm (same as q2_1 and q2_2) on the new dataset
    rand_idx = np.random.choice(self.N, self.k, replace=False)
    mu = self.X[:,rand_idx]
    Z = np.zeros(self.N, dtype=int)
    iter = 0
    while iter < 1000000:
      iter += 1
      Z_new = self.assign_clusters(self.X, Z, mu)
      mu_new = self.calc_cluster_means(self.X, Z_new, mu) 
      if np.array_equal(Z_new, Z):
        self.display(self.X0, Z, mu)
        break
      Z = Z_new
      mu = mu_new

  def assign_clusters(self, X, Z, mu):
    Z_new = Z.copy()
    for i in range(self.N):
      dist = []
      for j in range(self.k):
        dist.append(np.linalg.norm(X[:,i]-mu[:,j]))  
      Z_new[i] = dist.index(min(dist))
    return Z_new

  def calc_cluster_means(self, X, Z, mu):
    new_mu = mu.copy()
    for j in range(self.k):
      count = 0
      meanx = 0
      meany = 0
      for i in range(self.N):
        if Z[i] == j:
          count += 1
          meanx += X[0,i]
          meany += X[1,i]
      if count != 0:
        meanx = meanx/count
        meany = meany/count
      new_mu[0,j] = meanx
      new_mu[1,j] = meany
    return new_mu

  # Display results
  def display(self, X, Z, mu):
    colormap = {0:'darkorange',1:'blue',2:'lime',3:'cyan',4:'red'}
    fig, ax1 = plt.subplots(figsize=(5,5))

    # Cannot plot actual means of clusters here as they are vectors in higher dimensional space
    # The means in the 2 dimensional space has no significance in spectral clustering
    for i in range(self.N):
      ax1.scatter(self.X0[0,i], self.X0[1,i], color=colormap[Z[i]], alpha=0.2)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_xlim(self.xlim)
    ax1.set_ylim(self.ylim)
    ax1.set_aspect('equal')
    ax1.set_title(f'Number of Clusters = {self.k}, Kernel Param={self.param}')

    fig.tight_layout()
    fig.show()

# Running the algorithm on a specific random seed as the results of the algorithm depends
# heavily on the initialization done
# For example:
# seed 1, 101 -> good results for clustering with kernel B
# seed 2 -> bad results for clustering with kernel B
np.random.seed(1)
_clustering = Spectral_Clustering_1(data, 4, kernelB, 0.5)

# Kernel A doesn't perform as good as it doesn't reduce the data into a linear space
# _clustering = Spectral_Clustering_1(data, 4, kernelA, 2)

plt.show()



