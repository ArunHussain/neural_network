# **Implementation of perceptron and 3 layer neural network from scratch**

## **Perceptron**
A single neuron can be represnted as a perceptron and is the simplest neural network possible.
To get the output of the percepton, my implementation applies weights to each input and sums them and then uses an activation function to get the output.
To train the perceptron, the weight of each of the 2 inputs to it are adjusted by: *the error of the output \* the input \* the learning rate*

The issue with a perceptron is can it only solve linearly separable problems. Hence multilayer perceptrons are needed for more complex problems such as solving XOR outputs which is not linearly separable.


## **Three layer neural network**
The neural network implementation has a input, hidden and output layer and is fully connected. The output of the neural network is computed by a feed forward algorithm which first computes the output of the hidden layer and then using that, computes the output of the output layer. The formula is as follows:    
hidden_output = &sigma;(W<sub>ij</sub> * I + B)     
