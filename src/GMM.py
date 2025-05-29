import math
import numpy as np
from typing import Final
from k_means import K_Means as km
class GMM: # Mixture of Gaussians
    ITER: Final[int] = 1000
    def __init__(self , data , clusters):
        self.data = [np.array([data[i]]).transpose() for i in range(len(data))]
        self.dim = len(data[0])
        self.clusters = clusters
        model = km(data , clusters)
        model.train(GMM.ITER)
        self.mu = [np.array([[ j ] for j in i ]) for i in model.getCenters()]
        print(self.mu)
        self.phi = model.getProbs()
        self.cov = [np.eye(self.dim) for i in range(clusters)] # Todo - start with better cov :) -> calculate cov per cluster
    def getCenters(self):
        return self.mu
    def __str__(self):
        print(self.phi, end = '\n\n')
        print(self.mu , end = '\n\n')
        print(self.cov , end = '\n\n')
        return ""
    def normalPdf(self , x , ind):
        # print(x , ind)
        return math.exp(-1/2 * (x - self.mu[ind]).transpose() @ (np.linalg.inv(self.cov[ind]) @ (x-self.mu[ind])))/np.sqrt(np.linalg.det(self.cov[ind]) * (2 * math.pi)**len(x))
    def weight(self , x, ind):
        # Probability of ind given x
        return self.phi[ind] * self.normalPdf(x , ind)/sum(self.phi[i] * self.normalPdf(x , i) for i in range(self.clusters))
    def train(self , iter):
        for _ in range(iter):
            newCenters = [np.zeros((self.dim , 1)) for i in range(self.clusters)]
            newPhi = np.zeros((self.clusters , 1))
            for i in range(len(self.data)):
                for j in range(self.clusters):
                    w = self.weight(self.data[i] , j)
                    newPhi[j] += w
                    newCenters[j] += self.data[i] * w
            for i in range(self.clusters):
                newCenters[i]/=newPhi[i]
            newCov = [np.zeros((self.dim , self.dim)) for i in range(self.clusters)]
            for i in range(len(self.data)):
                for j in range(self.clusters):
                    w = self.weight(self.data[i] , j)
                    newCov[j] += w * (self.data[i] - newCenters[j]) @ (self.data[i] - newCenters[j]).transpose()
            for i in range(self.clusters):
                newCov[i]/=newPhi[i]
            newPhi/=len(self.data)
            self.mu = newCenters
            self.phi = newPhi
            self.cov = newCov
    def p(self , x):
        return sum(self.phi[i] * self.normalPdf(x , i) for i in range(self.clusters))