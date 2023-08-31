import numpy as np

def sigmoid(val): #this is an element wise function
   return 1/(1+np.exp(-val))
vectorised_sigmoid = np.vectorize(sigmoid)

def derivative_sigmoid(val): #This is an element wise function
    return val*(1-val) 
derivative_vectorised_sigmoid = np.vectorize(derivative_sigmoid)
 
class neural_network:

    def __init__ (self, num_input, num_hidden, num_output): #There are 3 layers     
        self.InputSize = num_input
        self.HiddenSize = num_hidden
        self.OutputSize = num_output 
        self.weights_inputs_to_hidden = np.random.randint(-1,1,(num_hidden, num_input))
        self.weights_hidden_to_output = np.random.randint(-1,1,(num_output, num_hidden))
        self.hidden_layer_bias = np.random.randint(-1,1,(num_hidden,1)) #this is a vector
        self.output_layer_bias = np.random.randint(-1,1,(num_output,1))
        self.learning_rate=0.01
     
    def feedForward(self, input): 
        #This function generates then returns the output of feeding in "input" to the neural network
        
        #Calculate output of hidden layer
        hidden_sums = np.matmul(self.weights_inputs_to_hidden, np.transpose(np.asmatrix(input)))
        hidden_sums = np.add(hidden_sums,self.hidden_layer_bias)
        hidden_output = vectorised_sigmoid(hidden_sums)

        #Calculate output of output layer
        output_sums = np.matmul(self.weights_hidden_to_output, hidden_output)
        output_sums = np.add(output_sums, self.output_layer_bias)
        output = vectorised_sigmoid(output_sums)

        return output
    
    def train(self, inputs, targets):
        #Code in "feedForward" is copied here as we need to access values it creates
        #Calculate output of hidden layer
        hidden_sums = np.matmul(self.weights_inputs_to_hidden, np.transpose(np.asmatrix(inputs)))
        hidden_sums = np.add(hidden_sums,self.hidden_layer_bias)
        hidden_output = vectorised_sigmoid(hidden_sums)

        #Calculate output of output layer
        output_sums = np.matmul(self.weights_hidden_to_output, hidden_output)
        output_sums = np.add(output_sums, self.output_layer_bias)
        outputs = vectorised_sigmoid(output_sums)

        #Calculate gradients of output
        output_errors = np.subtract(np.transpose(np.asmatrix(targets)),outputs) 
        output_gradients = derivative_vectorised_sigmoid(outputs)
        output_gradients = np.multiply(output_gradients,output_errors) #This is an element wise multiplication
        output_gradients = np.multiply(output_gradients,self.learning_rate)
        #Now we calculate deltas.
        hidden_output_transposed = np.transpose(hidden_output)
        hidden_output_deltas = np.matmul(output_gradients,hidden_output_transposed)
        #Next adjust weights and bias of output
        self.weights_hidden_to_output = np.add(self.weights_hidden_to_output, hidden_output_deltas) 
        self.output_layer_bias = np.add(self.output_layer_bias,output_gradients)

        #Calculate gradients of hidden
        weights_hidden_to_output_t = np.transpose(self.weights_hidden_to_output)
        hidden_errors=np.matmul(weights_hidden_to_output_t,output_errors)
        hidden_gradients = derivative_vectorised_sigmoid(hidden_output)
        hidden_gradients = np.multiply(hidden_gradients,hidden_errors)
        hidden_gradients = np.multiply(hidden_gradients, self.learning_rate)
        #Now we calculate deltas
        input_hidden_transposed = np.asmatrix(inputs)
        input_hidden_deltas = np.matmul(hidden_gradients,input_hidden_transposed)
        #Adjust weights and bias of hidden
        self.weights_inputs_to_hidden = np.add(self.weights_inputs_to_hidden,input_hidden_deltas)
        self.hidden_layer_bias = np.add(self.hidden_layer_bias, hidden_gradients)