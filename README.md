# **Implementation of perceptron and 3 layer neural network from scratch**

## **Perceptron**
A single neuron can be represnted as a perceptron and is the simplest neural network possible.
To get the output of the percepton, my implementation applies weights to each input and sums them and then uses an activation function to get the output.
To train the perceptron, the weight of each of the 2 inputs to it are adjusted by the error of output*the input*the learning_rate. 

The issue with a perceptron is can it only solve linearly separable problems, hence multilayer perceptrons are needed for more complex problems such as solving XOR outputs which is not lienarly separable.
