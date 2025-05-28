from k_means import K_Means as km
import numpy as np
import matplotlib.pyplot as plt

### Parameters
size = 1000
dim = 2
param = [[[-10 , -10] , np.eye(2)] , [[-5 , 5] , np.eye(2)] , [[5 , 5] , np.eye(2)] , [[5 , -5] , np.eye(2)] , [[0 , 0] , np.eye(2)] , [[-20 , 15] , np.eye(2)] , [[20 , 20] , np.eye(2)] , [[20 , -20] , np.eye(2)]]
###

k = [[np.random.default_rng().multivariate_normal(mean = mu , cov = cov) for i in range(size)] for (mu , cov) in param]

total = sum(k , [])
model = km(total , len(param))
model.train(10000)
centers = model.getCenters()

plt.scatter([i[0] for i in total] , [i[1] for i in total])

### Test
# colors = ['red' , 'brown' , 'purple' , 'yellow' , 'azure' , 'gold']
# for i in range(len(centers)):
#     plt.scatter([centers[i][0]] , [centers[i][1]] , s = 100 , c = colors[i])
###

plt.scatter([i[0] for i in centers] , [i[1] for i in centers] , c = 'red' , s = 100)
plt.xlim(-30 , 30)
plt.ylim(-30 , 30)
plt.show()

