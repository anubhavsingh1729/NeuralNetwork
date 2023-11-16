import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self,input_tensor):
        self.input_tensor = input_tensor-np.max(input_tensor)
        exp_inp = np.exp(self.input_tensor)
        exp_sum = np.sum(exp_inp,axis=1).reshape(-1,1)
        self.output_tensor = exp_inp/exp_sum
        return self.output_tensor
   
    def backward(self,error_tensor):
        return self.output_tensor*(error_tensor-(np.sum(error_tensor*self.output_tensor,axis=1)).reshape(-1, 1))
    