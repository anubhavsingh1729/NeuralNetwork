import copy

class NeuralNetwork:
    def __init__(self,optimizer):
        self.optimizer = optimizer
        self.loss=[]
        self.layers=[]
        self.data_layer=None
        self.loss_layer=None
        self.label_tensor = None
    
    def forward(self):
        input_tensor,label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        loss_output = self.loss_layer.forward(input_tensor,label_tensor)
        return loss_output
    
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)
    
    def append_layer(self,layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
    
    def train(self,iterations):
        for i in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)
    
    def test(self,input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        output = input_tensor
        return output