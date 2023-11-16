class BaseLayer:
    def __init__(self):
        self.trainable = False
    
    def forward(self,input_tensor):
        raise NotImplementedError("forward method not implemented")
        
    def backward(self,error_tensor):
        raise NotImplementedError("backward method not implemented")