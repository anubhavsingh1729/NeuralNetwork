from Layers.Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self,input_tensor):
        self.fx = np.maximum(0,input_tensor)
        return self.fx
    
    def backward(self,error_tensor):
        gradient = np.where(self.fx>0,error_tensor,0)
        return gradient
    