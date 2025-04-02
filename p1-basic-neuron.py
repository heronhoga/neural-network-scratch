inputs = [1.3, 2.8, 4.1]
weights = [5.6, 4.7, 2.2]
bias = 1.0 
output = 0.0

for i in range(len(inputs)):
    output += inputs[i] * weights[i]

output += bias

print(output)
