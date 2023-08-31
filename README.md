# **Implementation of perceptron and 3 layer neural network from scratch**

## **Perceptron**
A single neuron can be represnted as a perceptron and is the simplest neural network possible.
To get the output of the percepton, my implementation applies weights to each input and sums them and then uses an activation function to get the output.
To train the perceptron, the weight of each of the 2 inputs to it are adjusted by: *the error of the output \* the input \* the learning rate*

The issue with a perceptron is can it only solve linearly separable problems. Hence multilayer perceptrons are needed for more complex problems such as solving XOR outputs which is not linearly separable.


## **Three layer neural network**
The neural network implementation has a input, hidden and output layer and is fully connected. The output of the neural network is computed by a feed forward algorithm which first computes the output of the hidden layer and then using that, computes the output of the output layer. The formula of the algorithm is as follows:    

*H = &sigma;(W<sub>ij</sub> * I + B<sub>HL</sub>)*    
*O =  &sigma;(W<sub>ij</sub> * H + B<sub>OL</sub>)*     
*&sigma; is the activation function*
*H is the output of the hidden layer*  
*O is the output of the output layer*    
*W<sub>ij</sub> is the matrix of all weights in the network*     
*I is the inputs*    
*B<sub>HL</sub> is the bias of the hidden layer and B<sub>OL</sub> is the bias of the output layer*    

The training of the neural network has 7 key steps:
* Perform the feed forward algorithm
* Calculate the gradients of the output layer
* Calculate the deltas of the output layer
* Adjust the weights and bias of the output layer
* Calculate the gradients of the hidden layer
* Calcualte the deltas of the hidden layer
* Adjust the weights and bias of the hidden layer

The formula for calculating the weights and deltas of the output layer is:    
&delta;W<sub>ij<\sub><sup>HO</sup> = (lr * E * (O * (1-O))) . H<sup>T</sup>

The formula for calculating the weights and deltas of the hidden layer is:

