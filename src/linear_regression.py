import matplotlib.pyplot
import generalized_linear_model as glm
import pandas
import matplotlib

### Begin Hyper Parameters
learning_rate = .001
iterations = 1000
training_split = 10 # in percent
stochastic = 0 # Using SGD or BGD
### End Hyper Parameters

### Linear Regression
function = lambda hypothesis , input : sum([hypothesis[i] * input[i] for i in range(len(input))]) # straight line
guess = function
###


df = pandas.read_csv("Iris.csv")
df = df[df["Species"] == "Iris-setosa"]
df.drop(columns=["Id"])
matplotlib.pyplot.scatter(df["SepalLengthCm"].values.tolist() , df["SepalWidthCm"])
all_data = [[[df["SepalLengthCm"].values.tolist()[i]] , df["SepalWidthCm"].values.tolist()[i]] for i in range(len(df))]
linReg = glm.glm(function , guess , learning_rate , training_split , stochastic , iterations , all_data)
model = linReg.train()
loss = lambda guess, expected : (guess-expected)**2
print(linReg.testLoss(model , function , loss))
matplotlib.pyplot.plot([i for i in range(8)] , [guess(model , [1 , i]) for i in range(8)])
matplotlib.pyplot.show()
