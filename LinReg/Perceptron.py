import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.read_csv("Iris.csv")
all_input_features = [[1] + i for i in df.drop(columns = ["Id" , "Species"]).values.tolist()] # Insert constant item -> first term is constant
all_input_labels = [1 if i else -1 for i in df["Species"] == "Iris-setosa"] # Only linearly seperable flower -> Try to distinguish
all_input = list(zip(all_input_features , all_input_labels))

VAR = 5
TAU = 10
TRAINING_SIZE = 2 # in percent

theta = [0 for i in range(VAR)]
def diff(x , y): # >0 -> valid, else not valid
    global theta , VAR
    return sum([theta[i] * x[i] for i in range(VAR)]) * y
def update(x , y):
    global theta , VAR
    theta[0] += y
    for i in range(1 , VAR):
        theta[i] += x[i] * y

def train_perceptron(tau , training_input):
    global diff , theta
    updates = 0
    for i in range(tau):
        changed = False
        for [x , y] in training_input:
            if(diff(x , y) <= 0):
                update(x , y)
                updates += 1
                changed = True
        if(not changed):
            return updates
    return updates
def test_perceptron(testing_input):
    global diff
    cnt = len(testing_input)
    for [x , y] in testing_input:
        if(diff(x , y) <=0 ):
            cnt-=1
    print("Accuracy: %f" % (100*cnt/len(list(testing_input))))
def split_input(inp , p):
    num = p/100 * len(inp)
    num = int(num)
    f = random.sample(inp , num)
    s = [i for i in inp if i not in f]
    return [f , s]

training , testing = split_input(all_input , TRAINING_SIZE)
train_perceptron(TAU , training)
test_perceptron(testing)
