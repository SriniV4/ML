import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pandas as pd
import NN
import numpy as np
import math
import matplotlib.pyplot as plt

### Activation functions
sigmoid = lambda x : 1/(1+math.e**-x)
sigmoidGrad = lambda x : sigmoid(x) * (1-sigmoid(x))
relu = lambda x : x if x > 0 else 0
reluGrad = lambda x : 1 if x > 0 else 0
line = lambda x : x
lineGrad = lambda x :1
###

### Loss Functions
def crossEntropyGrad(guess , correct):
    S = sum(math.e**guess)
    return [math.e**guess[i]/S if i!=correct else -1 + math.e**guess[i]/S for i in range(len(guess))]
crossEntropyLoss = lambda guess , correct : -(guess[correct] - math.log((sum(math.e**guess))))
###

df = pd.read_csv("mnist_test.csv")
labels = df["label"].values
df = df.drop(columns = "label")
iter = 100
batch = 10
net = NN.NN([10] , 784 , [line] , [lineGrad])
for i in range(iter):
    for k in range(batch):
        net.forward_prop(np.array([[df.values[k][j]] for j in range(784)]))
        net.backward_prop(lambda x : crossEntropyGrad(x , labels[k]))
    net.updateNN()
for k in range(batch):
    net.forward_prop(np.array([[df.values[k][j]] for j in range(784)]))
    print(net.getGuess())
    print(labels[k]) 