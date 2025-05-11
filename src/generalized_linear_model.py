import matplotlib.pyplot as plt
import math
import random
class glm:
        def __init__(self , function , guess ,  learning_rate , training_size , stochastic, iterations , all_data):
                self.function = function # hypothesis function -> NOT ALWAYS THE PREDICTION ITSELF
                self.guess = guess
                self.learning_rate = learning_rate # step size
                self.training_size = int(training_size/100 * len(all_data)) # in percent
                self.stochastic = stochastic
                self.iterations = iterations
                all_data = [[[1] +i[0] ,i[1]] for i in all_data] # add garbage feature for constant term
                self.training = random.sample(all_data , self.training_size)
                self.testing = [i for i in all_data if i not in self.training]
                self.features = len(all_data[0][0])
        def train(self , weight = lambda input : 1):
                hypothesis = [0 for i in range(self.features)]
                for i in range(self.iterations):
                        if(self.stochastic):
                                for j in self.training:
                                        gradient = self.gradientStoch(j , hypothesis , weight)
                                        hypothesis = [hypothesis[i] + gradient[i] * self.learning_rate for i in range(len(gradient))]
                        else:
                                gradient = self.gradientBatch(hypothesis , weight)
                                hypothesis = [hypothesis[i] + gradient[i] * self.learning_rate for i in range(len(gradient))]
                return hypothesis
        def testLoss(self , hypothesis , function ,lossFunction):
                loss = 0
                for j in self.testing:
                        loss += lossFunction(function(hypothesis , j[0]) , j[1])
                return loss/len(self.testing)
        def testAccuracy(self , hypothesis): ## Classification problems
                correct = len(self.testing)
                for j in self.testing:
                        if(self.guess(hypothesis , j[0]) != j[1]):
                                correct -= 1
                return correct/len(self.testing)
        def testLocallyWeightedLoss(self , tau , function , lossFunction):
                loss = 0
                for (inp , out) in self.testing: 
                        weight = lambda input: math.exp(-sum([(inp[i] - input[i])**2 for i in range(len(input))])/(2 * tau**2))
                        hypothesis = self.train(weight)
                        ### Uncomment for default Iris code -> show line fitting each data
                        # plt.xlim(4 , 9)
                        # plt.ylim(1, 6)
                        # plt.scatter([a[0][1] for a in self.training + self.testing] , [a[1] for a in self.training+self.testing])
                        # plt.plot([i for i in range(4 , 8)] , [self.guess(hypothesis , [1 , i]) for i in range(4 , 8)])
                        # plt.show()
                        # print(hypothesis)
                        ###
                        loss += lossFunction(function(hypothesis , inp) , out)
                return loss/len(self.testing)
        def gradientBatch(self , hypothesis , weight): 
                # Assume valid input
                features = len(hypothesis)
                gradient = [0 for i in range(features)]
                m = len(self.training) # optional extra constant to normalize input -> calculate how much square difference on average 
                for (inp , out) in self.training:
                        for i in range(features):
                                gradient[i] += weight(inp) * inp[i] * (out - self.function(hypothesis , inp))
                # return gradient
                return [gradient[i]/m for i in range(features)]
        def gradientStoch(self ,  example, hypothesis , weight): # return gradient of single example
                return [weight(example[0]) * example[0][i] * (example[1] - self.function(hypothesis , example[0])) for i in range(len(hypothesis))]