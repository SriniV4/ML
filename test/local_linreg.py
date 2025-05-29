import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import matplotlib.pyplot
import generalized_linear_model as glm
import pandas
import matplotlib

### Begin Hyper Parameters
learning_rate = .001
iterations = 1000
training_split = 20 # in percent
stochastic = 1 # Using SGD or BGD
tau = .8 ## width of local -> higher tau -> local region is much bigger (quadratic inc/dec)
### End Hyper Parameters

### Linear Regression
function = lambda hypothesis , input : sum([hypothesis[i] * input[i] for i in range(len(input))]) # straight line
guess = function
###


df = pandas.read_csv("Iris.csv")
df.drop(columns=["Id"])
matplotlib.pyplot.scatter(df["SepalLengthCm"].values.tolist() , df["SepalWidthCm"])
all_data = [[[df["SepalLengthCm"].values.tolist()[i]] , df["SepalWidthCm"].values.tolist()[i]] for i in range(len(df))]
linReg = glm.glm(function , guess , learning_rate , training_split , stochastic , iterations , all_data)
loss = lambda guess, expected : (guess-expected)**2
print(linReg.testLocallyWeightedLoss(tau , function , loss))
