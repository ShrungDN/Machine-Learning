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

# Class for implementing Kernel PCA
class Kernel_PCA:
  def __init__(self, data, kernel, param, kernel_name):
    # X is data and N is number of data points
    # kernel is the kernel function the class uses
    # param is the perameter fed into kernel function
    self.X = data.copy()
    self.N = len(self.X[0])
    self.kernel = kernel
    self.param = param
    self.kernel_name = kernel_name

    # X and Y axis limits of plot
    self.xlim = np.array([0.9*np.min(self.X[0]), 1.1*np.max(self.X[0])])
    self.ylim = np.array([0.9*np.min(self.X[1]), 1.1*np.max(self.X[1])])

    # Centering the data
    Xs = self.X.sum(axis=1)
    Xs = np.expand_dims(Xs, axis=0)
    Xs = Xs.transpose()
    self.X = self.X - Xs/len(self.X[0])

    # Calculating the K-matrix using the kernel function
    self.K = np.zeros((self.N, self.N))
    for i in range(self.N):
      for j in range(self.N):
        self.K[i,j] = self.kernel(self.X[:,i], self.X[:,j], self.param)

    # Centering the kernel matrix (i.e centering the data in higher dimensional space)
    cmat = np.ones((self.N, self.N))/self.N 
    self.K = self.K - np.matmul(cmat, self.K) - np.matmul(self.K, cmat) + np.matmul(np.matmul(cmat,self.K), cmat)

    # Calculating top eigen values and corresponding eigen vectors of K-matrix (PCs in higher dimension)
    eigval, eigvec = np.linalg.eig(self.K)
    eigvec = eigvec.transpose()
    self.eig = list(zip(eigval, eigvec))
    self.eig.sort(reverse=True)
    self.topeig = self.eig[:2]
 
    # Calculating the projections of data in higher dimension onto the top 2 PCs
    self.projections = np.zeros((2, self.N))
    for k in range(2):
      lamk = self.topeig[k][0]
      betak = self.topeig[k][1]
      betak = betak/np.linalg.norm(betak) # Normalizing beta
      alphak = betak/np.sqrt(lamk)        # Scaling beta to form alpha
      alphak = alphak.real                # Using real part of alpha as imaginary part is negligible     
      for i in range(self.N):
        self.projections[k][i] = np.dot(self.K[i,:], alphak)

    # Displaying the plots
    self.display()
  
  def display(self):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'{self.kernel_name} with Parameter = {self.param}')
    ax.scatter(self.projections[0,:], self.projections[1,:])

for d in [2,3]:
  _polynomial_kpca = Kernel_PCA(data, kernelA, d, 'Kernel A')

for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
  _radial_basis_kpca = Kernel_PCA(data, kernelB, sigma, 'Kernel B')

plt.show()
