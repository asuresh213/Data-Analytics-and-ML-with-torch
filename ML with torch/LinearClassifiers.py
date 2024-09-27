import numpy as np
class Perceptron:
    def __init__(self,  eta=0.01, max_iter=70, seed=100):
        self.eta = eta
        self.max_iter = max_iter 
        self.seed = seed
    
    def predict(self, x):
        raw_output = np.dot(x, self.w) + self.b 
        return 1 if raw_output >= 0 else 0
    
    def train(self, inputs, labels):
        np.random.seed(self.seed)
        self.w = np.random.normal(0, 0.1, inputs.shape[1])
        self.b = np.random.rand()
        self.errors = []
        for i in range(self.max_iter):
            error = 0
            for x_i, label in zip(inputs, labels):
                dw = self.eta*(label - self.predict(x_i))*x_i
                db = self.eta*(label - self.predict(x_i))
                self.w += dw
                self.b += db
                error += int(db != 0.0) 
            self.errors.append(error)
        return 


# TO DO: Write a perceptron OvA function that will do classification over n classes. 


class Adaline:
    def __init__(self, opt = "GD", reg = "False", eta = 0.01, max_iter = 50, seed = 100, shuffle = 'True', activ = 'id', Lambda = 0.0):
        self.eta = eta
        self.max_iter = max_iter 
        self.seed = seed 
        self.opt = opt
        self.shuffle = shuffle
        self.activ = activ
        self.reg = reg 
        self.Lambda = Lambda

    def z_i(self, x):
        return self.activation(np.dot(x, self.w) + self.b)

    def activation(self, z):
        return z 
    
    def predict(self, x):
        return 1 if self.activation(self.z_i(x)) >= 0.5 else 0

    def shuffle_me(self, inputs, labels):
        perm = np.random.permutation(len(labels))
        return inputs[perm], labels[perm]

    def gradsGD(self, inputs, labels):
        n = inputs.shape[0]
        err = labels - self.z_i(inputs)
        loss = (self.activation(err)**2).mean()
        if(self.reg == True):
            loss += (self.Lambda/(2*n))*np.linalg.norm(self.w)**2
        grad_b = -2*err.mean()
        grad_w = -(2/n)*inputs.T.dot(err)
        if(self.reg == True):
            grad_w -= (self.Lambda/n)*self.w
        return grad_w, grad_b, loss
    
    def gradsSGD(self, x_i, label):
        err = label - self.z_i(x_i)
        loss = (self.activation(err)**2)

        if(self.reg == True):
            loss += (self.Lambda/(2))*np.linalg.norm(self.w)**2
        grad_w = (label - self.z_i(x_i))*x_i
        grad_b = label - self.z_i(x_i)
        if(self.reg == True):
            grad_w -= self.Lambda*self.w
        return grad_w, grad_b, loss 

    def train(self, inputs, labels):
        np.random.seed(self.seed)
        self.w = np.random.normal(0.0, 0.01, inputs.shape[1])
        self.b = np.random.rand()
        self.loss_vals = []
        
        if(self.shuffle == True):
            inputs, labels = self.shuffle_me(inputs, labels)

        if(self.opt == "GD"):
            
            for _ in range(self.max_iter):
                grad_w, grad_b, loss = self.gradsGD(inputs, labels)
                self.w += -self.eta*grad_w
                self.b += -self.eta*grad_b
                self.loss_vals.append(loss)
            return 
        
        elif(self.opt == "SGD"):
            for _ in range(self.max_iter):
                error = 0
                for x_i, label in zip(inputs, labels):
                    grad_w, grad_b, loss = self.gradsSGD(x_i, label)
                    self.w += self.eta*grad_w
                    self.b += self.eta*grad_b
                    error += loss 
                self.loss_vals.append(error/len(labels))
            return
        return
    

class LogisticRegression:
    def __init__(self, opt = "GD", reg = False, eta = 0.01, max_iter = 50, seed = 100, shuffle = 'True', activ = 'sigmoid', Lambda = 0.0):
        self.eta = eta 
        self.max_iter = max_iter 
        self.seed = seed 
        self.shuffle = shuffle 
        self.reg = reg
        self.opt = opt 
        self.activ = activ
        self.Lambda = Lambda
    
    def z_i(self, x):
        return np.dot(x, self.w) + self.b

    def activation(self, z):
        return 1/(1+np.exp(-np.clip(z,-100,100)))
    
    def gradsGD(self, inputs, labels):
        n = inputs.shape[0]
        sigma_z_i = self.activation(self.z_i(inputs))
        err = labels - sigma_z_i
        grad_w = inputs.T.dot(err)/n 
        grad_b = err.mean()
        if(np.allclose(sigma_z_i, labels)):
            loss = 0
            grad_w = np.zeros(inputs.shape[1])
            grad_b = 0
            return grad_w, grad_b, loss
        loss = (1/n)*(-labels.dot(np.log(sigma_z_i)) - (1-labels).dot(np.log(1-sigma_z_i)))
        if(self.reg == True):
            loss += (self.Lambda/(2*n))*(np.linalg.norm(self.w)**2)
            grad_w += + self.Lambda*self.w
        return grad_w, grad_b, loss

    
    def gradsSGD(self, x_i, label):
        sigma_z_i = self.activation(self.z_i(x_i))
        loss = -label*np.log(sigma_z_i) - (1-label)*np.log(1-sigma_z_i)
        grad_w = (label - sigma_z_i)*x_i
        grad_b = label - sigma_z_i
        if(self.reg == True):
            if(np.isclose(sigma_z_i, label)):
                loss = 0
                grad_w = np.zeros(len(x_i))
                grad_b = 0
                return grad_w, grad_b, loss
            loss += (self.Lambda/2)*(np.linalg.norm(self.w)**2)
            grad_w += self.Lambda*self.w
        return grad_w, grad_b, loss 
    
    def shuffle_me(self, inputs, labels):
        perm = np.random.permutation(len(labels))
        return inputs[perm], labels[perm]
    
    def train(self, inputs, labels):
        self.w = np.random.normal(0.0, 0.01, inputs.shape[1])
        self.b = np.random.rand()
        self.loss_vals = []
        self.params_initialized = True
        if self.shuffle == True:
            inputs, labels = self.shuffle_me(inputs, labels)
        if(self.opt == "GD"):
            for _ in range(self.max_iter):
                grad_w, grad_b, loss = self.gradsGD(inputs, labels)
                self.w += self.eta*grad_w
                self.b += self.eta*grad_b
                self.loss_vals.append(loss)
            return

        elif(self.opt == "SGD"):
            for _ in range(self.max_iter):
                error = 0
                for x_i, label in zip(inputs, labels):
                    grad_w, grad_b, loss = self.gradsSGD(x_i, label)
                    self.w += self.eta*grad_w
                    self.b += self.eta*grad_b
                    error += loss 
                self.loss_vals.append(loss)
            return
        

    def predict(self, x):
        return 1 if self.z_i(x) >= 0.5 else 0



