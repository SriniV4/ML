import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
import matplotlib.pyplot as plt
from GMM import GMM as gmm

### Parameters
size = 100
dim = 2
param = [[[-10 , -10] , np.eye(2)] , [[-5 , 5] , np.eye(2)] , [[5 , 5] , np.eye(2)] , [[5 , -5] , np.eye(2)] , [[0 , 0] , np.eye(2)] , [[-20 , 15] , np.eye(2)] , [[20 , 20] , np.eye(2)] , [[20 , -20] , np.eye(2)]]
###

k = [[np.random.default_rng().multivariate_normal(mean = mu , cov = cov) for i in range(size)] for (mu , cov) in param]
total = sum(k , [])
model = gmm(total , len(param))
model.train(100)
centers = model.mu
plt.scatter([i[0] for i in total] , [i[1] for i in total])
plt.scatter([center[0][0] for center in centers] , [center[1][0] for center in centers] , s = 100, color = 'red')
plt.show()
