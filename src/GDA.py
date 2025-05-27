import math
import numpy as np
class GDA:
    def __init__(self , positive_data, negative_data): # assume positive is label 1, negative is label 0 -> also assume same sigma
        self.m = len(positive_data) + len(negative_data)
        self.mu0 = 1/len(negative_data) * (sum([np.array([[negative_data[i][j]] for j in range(len(negative_data[i]))]) for i in range(len(negative_data))]))
        self.mu1 = 1/len(positive_data) * (sum([np.array([[positive_data[i][j]] for j in range(len(positive_data[i]))]) for i in range(len(positive_data))]))
        self.mu = [self.mu0, self.mu1]
        self.sigma0 = 1/self.m * sum([ ((np.array([[negative_data[i][j]] for j in range(len(negative_data[i]))]) - self.mu[0]) @ (np.array([[negative_data[i][j]] for j in range(len(negative_data[i]))]) - self.mu[0]).transpose() ) for i in range(len(negative_data)) ])
        self.sigma1 = 1/self.m * sum([ ((np.array([[positive_data[i][j]] for j in range(len(positive_data[i]))]) - self.mu[1]) @ (np.array([[positive_data[i][j]] for j in range(len(positive_data[i]))]) - self.mu[1]).transpose() ) for i in range(len(positive_data)) ])
        self.sigma = self.sigma0 + self.sigma1
        self.phi = len(positive_data)/self.m
    def __str__(self):
        print(self.mu , self.sigma , self.phi)
        return ""
    def normalPdf(self , x , ind):
        return math.exp(-1/2 * (x - self.mu[ind]).transpose() @ (np.linalg.inv(self.sigma) @ (x-self.mu[ind])))/math.sqrt((2*math.pi)**(len(x)) * np.linalg.det(self.sigma))
    def normalPdf1(self , x):
        return self.normalPdf(x , 1)
    def normalPdf0(self , x):
        return self.normalPdf(x , 0)
    def prob_positive(self):
        return self.phi
    def prob_negative(self):
        return 1 - self.phi
    def prob_x(self,  x):
        return self.prob_positive() * self.normalPdf1(x) + (self.prob_negative()) * self.normalPdf0(x)
    def prob_positive_given_x(self , x):
        return self.normalPdf(x , 1) * self.prob_positive() / self.prob_x(x)
    def prob_negative_given_x(self , x):
        return 1 - self.prob_positive_given_x(x)