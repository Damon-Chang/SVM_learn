import torch
import numpy as np
import matplotlib.pyplot as plt
import time

class SGDStruct:
    'SGD算法数据结构'
    def __init__(self, object_function, use_obj_fun, parameters):
    #def __init__(self, X_0, X, Y, loss_function, object_function, use_obj_fun, parameters):
        if use_obj_fun != True:
            self.X = X    #样本
            self.Y = Y    #标签
            self.loss = loss_function
        self.parameters = parameters
        self.X_0 = parameters['beginning_point']
        self.X_k = self.X_0
        self.epsilon = parameters['threshold']
        
    def Loss(self):
        if use_obj_fun:
            return object_function(self.X_k[0],self.X_k[1])
        else:
            return self.loss(self.X, self.Y)

def object_function(X, Y):
    """
    INPUT：自变量X（向量）
    OUTPUT：目标方程
    """
    return 10*X**2 + Y**2

def MSE(X, Y):
    #return np.mean((torch.tensor(X) - torch.tensor(Y))**2)
    return np.mean(np.square(np.array(X - Y)))

def BB_step(s, y):
    """
    INPUT:两个二维向量，存储当前位置X[1]和上一个位置X[0]的迭代点和梯度值
    OUTPUT:两个BB步长
    """
    #s = X[1] - X[0]
    #y = grad[1] -grad[0]
    BB1 = s.dot(s) / s.dot(y)
    BB2 = s.dot(y) / y.dot(y)
    return BB1, BB2


def train():
    X_k_1 = torch.tensor([0.,0.]).requires_grad_(True)
    grad_k_1 = torch.tensor([0.,0.])
    X_dif = torch.tensor([0.,0.])
    grad_dif = torch.tensor([0.,0.])
    X_k = torch.tensor([0.,0.]).requires_grad_(True)
    X_k_Seq = []
    
    K = SGD.parameters['max_iteration']
    
    for i in range(K):
        if i == 0:
            SGD.X_k = SGD.X_0.requires_grad_(True)
            print(SGD.X_k)
            X_k_Seq.append(SGD.X_k)
            y = object_function(SGD.X_k[0],SGD.X_k[1])
            y.backward()
            
            X_k = SGD.X_k
            grad_k = SGD.X_k.grad
            print(grad_k)
            
            diff = MSE(grad_k, torch.tensor([0.,0.]))
            
            grad_dif = grad_k
            X_dif = X_k
            
            eta = BB_step(X_dif, grad_dif)[0]
            SGD.X_k = SGD.X_k - eta * grad_k
            print(i,'th X__k:',SGD.X_k)
            X_k_Seq.append(SGD.X_k)
            
            grad_k_1 = grad_k
            X_k_1 = X_k
            
        else:
            #计算当前函数值，梯度
            X_k = torch.tensor(np.array(SGD.X_k.detach())).requires_grad_(True)
            print(X_k)
            y = object_function(X_k[0],X_k[1])
            print(y)
            y.backward()
            
            #计算梯度
            print(X_k)
            grad_k = X_k.grad
            print(grad_k)
            
            diff = MSE(grad_k, torch.tensor([0.,0.]))

            #更新，计算下一步迭代点
            eta = BB_step(X_dif, grad_dif)[0]
            SGD.X_k = X_k - eta * grad_k
            print(i,'th X__k:',SGD.X_k)
            X_k_Seq.append(SGD.X_k)

            #更新差值，为了计算BB步长
            grad_dif = grad_k - grad_k_1
            X_dif = X_k - X_k_1

            #存储当前梯度和迭代点
            grad_k_1 = grad_k
            X_k_1 = X_k
        
        
        if diff <= SGD.epsilon:
            break
            
    return X_k, X_k_Seq, grad_k_1

def train_SGD():
    
    eta = 0.01
    
    X_k = torch.tensor([0.,0.]).requires_grad_(True)
    X_k_Seq = []
    
    K = SGD.parameters['max_iteration']
    
    for i in range(K):
        if i == 0:
            SGD.X_k = torch.tensor([10.,-10.]).requires_grad_(True)
            #print(SGD.X_k)
            X_k_Seq.append(SGD.X_k)
            y = object_function(SGD.X_k[0],SGD.X_k[1])
            y.backward()
            
            X_k = SGD.X_k
            grad_k = SGD.X_k.grad
            #print(grad_k)
            
            diff = MSE(grad_k, torch.tensor([0.,0.]))
            
            SGD.X_k = SGD.X_k - eta * grad_k
            #print(i,'th X__k:',SGD.X_k)
            X_k_Seq.append(SGD.X_k)

            print('Ieration:', i)
            
        else:
            X_k = torch.tensor(np.array(SGD.X_k.detach())).requires_grad_(True)
            #print(X_k)
            y = object_function(X_k[0],X_k[1])
            #print(y)
            y.backward()
            
            
            #print(X_k)
            grad_k = X_k.grad
            #print(grad_k)
            
            diff = MSE(grad_k, torch.tensor([0.,0.]))

            SGD.X_k = X_k - eta * grad_k
            #print(i,'th X__k:',SGD.X_k)
            X_k_Seq.append(SGD.X_k)

            print('Ieration:', i)

        if diff <= SGD.epsilon:
            break
            
    return X_k, X_k_Seq

def show(x_k, x_k_1):
    y_ax = np.linspace(-25.,100.,20)
    x_ax = np.linspace(-200.,200.,20)
    X = torch.tensor(np.stack((x_ax,y_ax), axis = 1))
    x,y = np.meshgrid(X[:,0], X[:,1])

    plt.contour(x, y, object_function(x,y), linestyles = 'dashed')
    plt.plot(np.array(x_k)[:,0], np.array(x_k)[:,1], '-rs')
    plt.plot(np.array(x_k_1)[:,0], np.array(x_k_1)[:,1], '-b.')
    plt.title("$ f(x,y)=10x^2+y^2 $")
    plt.xlabel("$ x $")
    plt.ylabel("$ y $")
    #plt.savefig(r'D:\a学业信计\a研究生\工作笔记\周总结\pic'+'\\1.svg')
    plt.show()

if __name__ == '__main__':
    parameters = {
    'beginning_point':torch.tensor([10.,30.]).requires_grad_(True),
    'threshold':1e-5,
    'max_iteration':3000,
    }
    
    SGD = SGDStruct(object_function, True, parameters)
    SGD.X_0 = parameters['beginning_point']
    X_k_final = train()
    x_k = [list(i.detach()) for i in X_k_final[1]]
    x_sgd = train_SGD()
    x_k_1 = [list(i.detach()) for i in x_sgd[1]]
    show(x_k, x_k_1)

