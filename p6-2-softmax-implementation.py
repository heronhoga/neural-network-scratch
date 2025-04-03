import numpy as np
from create_data import createData
np.random.seed(0)

X, y = createData(100, 3)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
print(X.shape)

# layer1 = LayerDense(4, 5) #input = 4 neurons, output = 5 neurons
# layer2 = LayerDense(5, 2) #input = 5 neurons, output = 2 neurons
# layer1 = LayerDense(2, 5) #input = 2 neurons, output = 5 neurons
# activation1 = ActivationReLU()

# layer1.forward(X)
# activation1.forward(layer1.output)
# print(activation1.output)

