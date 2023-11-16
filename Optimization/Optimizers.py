class Sgd:
    def __init__(self,learning_rate):
        self.lr = learning_rate
        
    def calculate_update(self,weight_tensor, gradient_tensor):
        up_weight = weight_tensor - self.lr*gradient_tensor
        
        return up_weight
    