import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5], 
          [2.0, 5.0, -1.0,2.0],
          [-1.5, 2.7, 3.3, -0.8]
          ]

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = LayerDense(4, 5) #input = 4 neurons, output = 5 neurons
layer2 = LayerDense(5, 2) #input = 5 neurons, output = 2 neurons

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)

    
# weights = [
#     [0.2, 0.8, -0.5, 1.0],  
#     [0.5, -0.91, 0.26, -0.5], 
#     [-0.26, -0.27, 0.17, 0.87]
# ]

# biases = [2,3,0.5] 

# weights2 = [
#     [0.1, -0.14, 0.5],  
#     [-0.5, 0.12, -0.33], 
#     [-0.44, 0.73, -0.13]
# ]

# biases2 = [-1,2, -0.5] 

# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2


# print(layer2_outputs)
