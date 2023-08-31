import random 
import math
import numpy

class perceptron: #Supervised learning perceptron
    weights=[]
    learning_rate=0.1
    def __init__(self):
        for i in range(0,2):
            self.weights.append(round(random.uniform(-1,1),1)) #intialising 2 weights randomly
    
    def  output(self, inputs): #Shows the output of perceptron
        sum=0
        for i in range(0,len(self.weights)):
            sum = sum+self.weights[i]*inputs[i]
        output = (-1) if sum<0 else 1 #activation function
        return output
    
    def train(self, inputs, target): #this takes inputs (a single x,y pair) as well as a target
        current_output = self.output(inputs) 
        error = target - current_output
        for i in range (0,len(self.weights)):
            self.weights[i]+=error*inputs[i]*self.learning_rate     
