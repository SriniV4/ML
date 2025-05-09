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
        def train(self):
                hypothesis = [0 for i in range(self.features)]
                for i in range(self.iterations):
                        if(self.stochastic):
                                for j in self.training:
                                        gradient = self.gradientStoch(j , hypothesis)
                                        hypothesis = [hypothesis[i] + gradient[i] * self.learning_rate for i in range(len(gradient))]
                        else:
                                gradient = self.gradientBatch(hypothesis)
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
        def gradientBatch(self , hypothesis): 
                # Assume valid input
                features = len(hypothesis)
                gradient = [0 for i in range(features)]
                m = len(self.training) # optional extra constant to normalize input -> calculate how much square difference on average 
                for (inp , out) in self.training:
                        for i in range(features):
                                gradient[i] += inp[i] * (out - self.function(hypothesis , inp))
                # return gradient
                return [gradient[i]/m for i in range(features)]
        def gradientStoch(self ,  example, hypothesis): # return gradient of single example
                return [example[0][i] * (example[1] - self.function(hypothesis , example[0])) for i in range(len(hypothesis))]

