import math
import random
import numpy as np
class Layer:
    def __init__(self , prev , neurons , activation , activationGrad): # layer contains weights from previous layer to current layer as matrix
        self.neurons = neurons
        self.activation = activation
        self.activationGrad = activationGrad
        self.prev = prev
        # neurons x prev matrix
        self.weights = np.array([[random.random()/neurons**4 for i in range(prev)]for i in range(neurons)])
        # self.weights = np.array([[j for i in range(prev)] for j in range(neurons)])
        self.bias = np.array([[random.random()/neurons**4] for i in range(neurons)])
        self.zero_grad()
    def feed_forward(self , input):
        assert(len(input) == self.prev)
        self.x = np.array(input)
        self.z = (self.weights @ self.x) + self.bias
        self.a = self.activation(self.z)
    def get_forward_output(self):
        return self.a
    def get_backward_output(self):
        return self.outputGrad
    def __str__(self):
        print(self.weights)
        print(self.bias)
        print()
        return ""
    def zero_grad(self):
        self.weightGrad = np.array([[0. for i in range(self.prev)] for i in range(self.neurons)])
        self.biasGrad = np.array([[0.] for i in range(self.neurons)])
    def update(self , learning_rate): # gradient descent (sub gradient)
        self.weights -= self.weightGrad * learning_rate
        self.bias -= self.biasGrad * learning_rate
    def feed_backward(self , input): # input is partials with respect to this layer
        tempGrad = input * self.activationGrad(self.z)
        # print(np.vstack([self.x.transpose() for i in range(self.neurons)]) * tempGrad)
        self.biasGrad += tempGrad
        self.weightGrad += np.vstack([self.x.transpose() for i in range(self.neurons)]) * tempGrad
        # print(np.array([[sum([self.weights[i][j] * tempGrad[i][0] for i in range(self.neurons)])] for j in range(self.prev)]))
        self.outputGrad = np.array([[sum([self.weights[i][j] * tempGrad[i][0]  for i in range(self.neurons)])] for j in range(self.prev)])
class NN:
    def __init__(self , layers , input , activation , activationGrad): # input = dim(input)
        assert(len(layers))
        self.layers = [Layer(layers[i-1] if i else input ,layers[i] , activation[i] , activationGrad[i]) for i in range(len(layers))]
    def forward_prop(self , input):
        for i in range(len(self.layers)):
            self.layers[i].feed_forward(self.layers[i-1].get_forward_output() if i else input)
    def backward_prop(self , lossGrad):
        for i in range(len(self.layers)-1 , -1 , -1):
            self.layers[i].feed_backward(self.layers[i+1].get_backward_output() if i != len(self.layers)-1 else lossGrad(self.layers[-1].get_forward_output()))
    def updateNN(self):
        for i in self.layers:
            i.update(.00001)
            i.zero_grad()
    def getGuess(self):
        return self.layers[-1].get_forward_output()
