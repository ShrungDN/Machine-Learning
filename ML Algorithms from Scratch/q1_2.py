import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the csv file
df = pd.read_csv('Dataset.csv', header=None)
data = df.to_numpy()
data = data.transpose()

# Class for implementing PCA without data centering
class PCA2:
  def __init__(self, data):
    # X is data and N is number of data points
    self.X = data.copy()
    self.N = len(self.X[0])

    # X and Y axis limits of plot
    self.xlim = np.array([np.min(self.X[0])-2, np.max(self.X[0]+2)])
    self.ylim = np.array([np.min(self.X[1])-2, np.max(self.X[1]+2)])

    # No centering of data here

    # Calculating top eigen values and corresponding eigen vectors (PCs)
    K = np.matmul(self.X, self.X.transpose())/self.N
    eigval, eigvec = np.linalg.eig(K)
    eigvec = eigvec.transpose()
    self.eig = list(zip(eigval, eigvec))
    self.eig.sort(reverse=True)

    # Calculating results
    self.mean = self.X.sum(axis=1)/self.N
    
    # Variance of projected lenghts is same as the eigen value for centered data, hence:
    var1 = 0
    var2 = 0
    for i in range(self.N):
      var1 += (np.dot(self.X[:, i], self.eig[0][1]) - self.mean[0])**2
      var2 += (np.dot(self.X[:, i], self.eig[1][1]) - self.mean[1])**2
    self.variance1 = var1/self.N
    self.variance2 = var2/self.N
    self.percent_variance = np.array([self.variance1, self.variance2])/(self.variance1 + self.variance2)*100

    # Error (or residue) with respect to each PC
    err1 = 0
    err2 = 0
    for i in range(self.N):
        err1 += (np.linalg.norm(self.X[:, i]))**2 - (np.dot(self.X[:, i], self.eig[0][1]))**2
        err2 += (np.linalg.norm(self.X[:, i]))**2 - (np.dot(self.X[:, i], self.eig[1][1]))**2
    self.err1 = err1/self.N
    self.err2 = err2/self.N
    
    # Display the results
    self.display()
  
  def display(self):
    print('Mean of Data is = ', self.mean)
    print('[Eiegen Value, Eigen Vector] pairs are:', self.eig)
    print('Variance of projected lengths along PC1 = ', self.variance1)
    print('Variance of projected lengths along PC2 = ', self.variance2)
    for i in range(2):
      print(f'Percentage of Variance explained by PC{i+1} = ', self.percent_variance[i])  
    print('Error or Residue with respect to PC1 = ', self.err1) 
    print('Error or Residue with respect to PC2 = ', self.err2)  

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,10))
    ax1.set_xlim(self.xlim)
    ax1.set_ylim(self.ylim)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title(f'Data and Top 2 Principal Components')
    ax1.scatter(self.X[0,:], self.X[1,:], label='Data', color='springgreen')
    for i, (_, pc) in enumerate(self.eig):
      ax1.plot(self.xlim, (pc[0]/pc[1])*self.xlim, label=f'PC{i+1}')
    ax1.set_aspect('equal')
    ax1.legend(loc='best')

    pc_names = ['PC1', 'PC2']
    explode = [0.01, 0.01]
    ax2.pie(self.percent_variance, labels=pc_names, autopct='%1.3f%%', explode=explode)
    ax2.set_title('Percentage of Variance explained by Principal Components')

pca2 = PCA2(data)
plt.show()