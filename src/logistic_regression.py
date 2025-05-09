import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import generalized_linear_model as glm

### Begin Hyper Parameters
learning_rate = .01
training_split = 20 # in percent
stochastic = 1
iterations = 5000
###

### Functions
def sigmoid(x):
    return 1/(1+math.exp(-x))
function = lambda hypothesis , input : sigmoid(sum([hypothesis[i] * input[i] for i in range(len(input))]))
guess = lambda hypothesis , input : 1 if function(hypothesis , input) >= .5 else 0 # deterministic guess
# guess = lambda hypothesis , input : 1 if random.random() <= function(hypothesis , input) else 0 # probabilistic guess
###

df = pd.read_csv("Iris.csv")
df["Species"] = [0 if df["Species"][i] == "Iris-setosa" else 1 for i in range(len(df))]
all_data = [[df.drop(columns = ["Species"]).values.tolist()[i] , df["Species"].values.tolist()[i]] for i in range(len(df))]
logReg = glm.glm(function , guess , learning_rate , training_split , stochastic , iterations , all_data)
model = logReg.train()
loss = lambda guess , correct : correct * (-1e10 if not guess else math.log(guess)) + (1-correct)*(-1e10 if not (1-guess) else (math.log(1 - guess)))
print(logReg.testAccuracy(model))
print(logReg.testLoss(model , function , loss))