import GDA
import numpy as np
import random

### Params
d = 100 # dimensions
mu0 = [0 for i in range(d)] # mean of gaussian with label 0
iter0 = 1000
mu1 = [3 for i in range(d)] # mean of gaussian with label 1
iter1 = 1000
s0 = [[10 if i == j else 0 for j in range(d)] for i in range(d)] # covariance matrix of gaussian label 0
s1 = [[2 if i == j else 0 for j in range(d)] for i in range(d)] # cov matrix of guassian label 1
###

positive_data = [np.random.default_rng().multivariate_normal(mu1 , s1) for i in range(iter0)]
negative_data = [np.random.default_rng().multivariate_normal(mu0 , s0) for i in range(iter1)]
model = GDA.GDA(positive_data , negative_data)
print(model)
test_positive_data = [np.random.default_rng().multivariate_normal(mu1 , s1) for i in range(iter0)]
test_negative_data = [np.random.default_rng().multivariate_normal(mu0 , s0) for i in range(iter1)]
for i in test_positive_data + test_negative_data:
    temp = [[j] for j in i]
    print(model.prob_positive_given_x(temp))