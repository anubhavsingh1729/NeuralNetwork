import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass
    
    def forward(self,prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        eps = np.finfo(float).eps
        prediction_tensor = prediction_tensor+eps
        loss = -np.sum(label_tensor*np.log(prediction_tensor))
        #print(loss)
        return loss
    
    def backward(self,label_tensor):
        error_tensor = np.divide(-label_tensor,self.prediction_tensor)
        return error_tensor