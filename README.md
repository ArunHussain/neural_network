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
* Calculate the gradients of the hidden to output connections   
* Calculate the deltas of the hiddent to output connections    
* Adjust the weights and bias of the output layer by the deltas and gradients repectively
* Calculate the gradients of the input to hidden connections   
* Calculate the deltas of the input to hidden connections   
* Adjust the weights and bias of the hidden layer by the deltas and gradients respectively

The formula for calculating the deltas of the connections from the hidden to output layer is:    
*&delta;W<sub>ij</sub><sup>HO</sup> = (lr * E * (O * (1-O))) . H<sup>T</sup>*   
*(lr * E * (O * (1-O))) is the gradient*

The formula for calculating the deltas of the connections from the input to hidden layer is:   
*&delta;W<sub>ij</sub><sup>IH</sup> = (lr * HE * (H * (1-H))) . I<sup>T</sup>*   
*(lr * HE * (H * (1-H))) is the gradent*   

*&delta;W<sub>ij</sub><sup>HO</sup> is the change in the weights of the connections from the hidden to output layer*     
*&delta;W<sub>ij</sub><sup>IH</sup> is the change in the weights of the connections from the input to the hidden layer*   
*lr is the learning rate*   
*E is the errors of the output layer*   
*HE is the errors of the hidden layer*    
*O is the outputs of the output layer*     
*H is the outputs of the hidden layer*   
*H<sup>T</sup> is the transpose of H*    
*I<sup>T</sup> is the transpose of the inputs to the input layer*    
*\* is the Hadamard product (i.e the element wise multiplication)*   
*. is the matrix product*

