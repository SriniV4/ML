import numpy as np
import math
import random
class K_Means:
    def __init__(self , data , k):
        self.data = data
        self.k = k
        self.dim = len(data[0])
        assert(k <= len(data))
        # self.centers = random.sample(data , k) # Initialize clusters randomly -> performs poorly
        # kmeans ++ -> initialize with point farthest from closest center
        self.centers = random.sample(data , 1)
        distance = lambda x , y : np.sqrt(np.sum([abs(y[i] - x[i]) ** 2 for i in range(len(x))]))
        while(len(self.centers) != self.k):
            dist = []
            for j in data:
                dist.append(min([distance(j , i) for i in self.centers]))
            self.centers.append(data[np.argmax(dist)])
            # print(self.centers)

    def train(self , iterations):
        for _ in range(iterations):
            self.newCenters = np.zeros((self.k, self.dim))
            self.newCounts = np.zeros((self.k , 1))
            for j in self.data:
                mnDist = math.inf
                mnInd = -1
                for i in range(self.k):
                    dist = math.dist(self.centers[i] , j)
                    if(dist < mnDist):
                        mnDist = dist
                        mnInd = i
                self.newCenters[mnInd] += np.array(j)
                self.newCounts[mnInd]+=1
            self.newCenters/=self.newCounts
            if(np.array((self.centers == self.newCenters)).all()):
                break
            self.centers = self.newCenters
            self.counts = self.newCounts
    def getCenters(self):
        return self.centers
    def getProbs(self):
        return self.counts / np.array(len(self.data))
    
                    


