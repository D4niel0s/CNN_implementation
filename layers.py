from abc import ABC, abstractmethod
import cupy as cp
from funcs import *


class Layer(ABC):
    def __init__(self):
        self.input = None;
        self.output = None;

    @abstractmethod
    def forward(self, inp, batch_size=1):
        pass
    
    @abstractmethod
    def backward(self, upstream_grad, learning_rate, batch_size=1):
        pass


class Fully_connected(Layer):
    def __init__(self, input_dim, output_dim):
        self.W = cp.random.uniform(0,1/input_dim ,(output_dim, input_dim)) #Kiaming initialization
        self.b = cp.random.random((output_dim, 1))

    def forward(self, inp, batch_size=1):
        self.input = inp
        self.output = (self.W @ inp) + self.b

        return self.output

    def backward(self, upstream_grad, learning_rate, batch_size=1):
        #Assume upstream is a matrix where each column is a sample's upstream gradient, self.input is matrix of order dxn. (d-dimension of point, n-batch size)
        dW = (upstream_grad @ self.input.T)/batch_size
        db = cp.mean(upstream_grad, axis=1, keepdims=True)
        dx = self.W.T @ upstream_grad

        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dx
    

class Activation(Layer):
    def __init__(self, func, func_derivative):
        self. activation = func
        self. derivative = func_derivative

    def forward(self, inp, batch_size=1):
        self.input = inp
        self.output = self.activation(inp)

        return self.output
    
    def backward(self, upstream_grad, learning_rate, batch_size=1):
        return upstream_grad * self.derivative(self.input)
    

class ReLU(Activation):
    def __init__(self):
        super().__init__(self, relu, relu_derivative) #Defined in funcs