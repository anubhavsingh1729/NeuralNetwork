from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0,1,(input_size+1,output_size))
        self._optimizer = None
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self,val):
        self._optimizer = val
    
    def forward(self, input_tensor):
        #self.input_tensor = input_tensor
        self.input_tensor = np.append(input_tensor, np.ones((len(input_tensor), 1)), axis=1)
        #print(input_tensor.shape,self.weights.shape)
        return np.dot(self.input_tensor,self.weights)
    
    def backward(self, error_tensor):
        gradient = np.dot(error_tensor,self.weights[:-1].T)
        self.gradient_weights =  np.dot(self.input_tensor.T,error_tensor)
        
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights,self.gradient_weights)
        return gradient
    
