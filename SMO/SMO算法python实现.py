# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:15:37 2022

@author: ASUS-PC
"""
"""
参考文献：Sequential minimal optimization: A fast algorithm for training support vector machines
作者：John Platt
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
#%%
class SMOStruct:
    '''构造SMO算法数据结构'''
    def __init__(self, X, y, C, kernel, alphas, b, errors, user_linear_optim):
        self.X = X              #训练样本
        self.y = y              #类别标签
        self.C = C              #regularization parameter 正则化常量
        self.kernel = kernel    #kernel function 核函数，实现了两个和函数：线性核函数和高斯径向基（RBF）
        self.alphas = alphas    #lagrange multiplier 拉格朗日乘子
        self.b = b              #scalar bias term 标量，表示偏移量
        self.error = errors     #error cache 用于存储alpha值实际与预测值的差值
        
        self.m, self.n = np.shape(self.X)   #样本个数m，和每个样本特征数量n
        self.user_linear_optim = user_linear_optim    #模型是否使用线性核函数
        self.w = np.zeros(self.n)   #初始化权重w的值，主要用于线性核函数
      
    def predict(self, x):
        if model.user_linear_optim:
            return round(float(self.w.T @ x) - self.b)
        else:
            #根据SVM算法中的最终解出来的超平面的表达式
            return round(np.sum([self.alphas[j] * self.y[j] * self.kernel(x, model.X[j]) for j in range(model.m)]) - model.b)

      
def linear_kernel(x, y, b = 1):
    '''线性核函数，返回矩阵x和y的线性组合，初始化偏执b=1'''
    result = x @ y.T + b #其中@运算符表示矩阵相乘
    return result 

def gaussian_kernel(x, y, sigma = 1):
    
    '''高斯核函数，返回序列x，y的相似性，核函数中的参数是sigma'''
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(-(np.linalg.norm(x - y, 2))**2 / (2 * sigma**2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(-(np.linalg.norm(x - y, 2, axis = 1))**2 / (2 * sigma**2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(-(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis = 2) ** 2) / (2*sigma**2))
    
    return result

#判别函数一，用于单一样本
def decision_function_output(model, i):
    if model.user_linear_optim:
        return float(model.w.T @ model.X[i]) - model.b
    else:
        #根据SVM算法中的最终解出来的超平面的表达式
        return np.sum([model.alphas[j] * model.y[j] * model.kernel(model.X[i], model.X[j]) for j in range(model.m)]) - model.b
   
#判别函数二，用于多个样本
def decision_function(alphas, target, kernel, X_train, x_test, b):
    '''对输入样本特征使用SVM判别函数判别样本所属类别'''
    result = (alphas * target) @ kernel(X_train, x_test) - b
    return result

def plot_decision_boundary(model, ax, resolution = 100, colors = ('b', 'k', 'r'), levels = (-1, 0, 1)):
    """
    绘制分割平面以及支持平面，使用等高线方法，评估支持平面的准确性
    """
    xrange = np.linspace(model.X[:,0].min(), model.X[:,0].max(), resolution)
    yrange = np.linspace(model.X[:,1].min(), model.X[:,1].max(), resolution)
    grid = [[decision_function(model.alphas, model.y, model.kernel, model.X, np.array([xr,yr]),model.b) for xr in xrange] for yr in yrange]
    grid = np.array(grid).reshape(len(xrange), len(yrange))#返回一个矩阵，m*m
    
    #用散点图绘制决策轮廓
    #为训练数据绘制散点图
    ax.contour(xrange, yrange, grid, levels = levels, linewidths = (1,1,1), linestyles = ('--', '-', '--'), colors = colors)
    ax.scatter(model.X[:,0], model.X[:,1], c = model.y, cmap = plt.cm.viridis, lw = 0, alpha = 0.25)
    #viridis()函数用于将颜色映射设置为“viridis”
    #绘制支持向量
    mask = np.round(model.alphas, decimals = 2) != 0.0#np.round(数据, decimal=保留的小数位数)
    #mask对应乘子不等于零的样本，也就是支持向量。
    ax.scatter(model.X[mask,0], model.X[mask,1], c = model.y[mask], cmap = plt.cm.viridis, lw = 1, edgecolors = 'k')
    
    return grid, ax
#%%
'''--------------外循环和内循环选择两个训练样本-------------'''
def get_error(model, i):
    if 0 < model.alphas[i] < model.C:
        return model.errors[i]
    else:
        return decision_function_output(model, i) - model.y[i]

def take_step(i1, i2, model):
    #如果选择两个相同的样本则跳过该步骤
    if i1 == i2:
        return 0, model
    #alpha2,alpha2原始值和更新值表示都和论文中的一致
    alpha1 = model.alphas[i1]
    alpha2 = model.alphas[i2]
    
    y1 = model.y[i1]
    y2 = model.y[i2]
    
    E1 = get_error(model, i1)
    E2 = get_error(model, i2)
    
    s = y1 * y2
    
    #计算alpha得边界L,H
    if (y1 != y2):
        #y1,y1异号
        L = max(0, alpha2 - alpha1)
        H = min(model.C, model.C + alpha2 - alpha1)
    elif (y1 == y2):
        #y1,y2同号
        L = max(0, alpha1 + alpha2 - model.C)
        H = min(model.C, alpha1 + alpha2)
    if (L == H):
        return 0, model
    
    #计算和函数组合
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])    
    
    #计算eta，即alpha2更新方向
    eta = k11 + k22 - 2*k12
    
    #一般情况下eta>0。根据eta正负对alpha2做更新
    if eta > 0:
        alpha2_new = alpha2 + y2 * (E1 - E2) / eta
        #将alpha剪到限定区间内
        if L < alpha2_new < H:
            alpha2_up = alpha2_new
        elif alpha2_new <= L:
            alpha2_up = L
        elif alpha2_new >= H:
            alpha2_up = H
    else:
        #特殊情况下，eta取值有可能为负
        f1 = y1 * (E1 + model.b) - alpha1 * k11 - s * alpha2 * k12
        f2 = y2 * (E2 + model.b) - s * alpha1 * k22 - alpha2 * k22
        
        L1 = alpha1 + s * (alpha2 - L)
        H1 = alpha1 + s * (alpha2 - H)
        
        L_obj = L1 * f1 + L * f2 + 0.5 * (L1**2) * k11 + 0.5 * (L**2) * k22 + s * L * L1 * k12
        H_obj = H1 + f1 + H * f2 + 0.5 * (H1**2) * k11 + 0.5 * (H**2) * k22 + s * H * H1 * k12
            
        if L_obj < H_obj - eps:#左边界小于右边界
            alpha2_up = L
        elif L_obj > H_obj + eps:
            alpha2_up = H
        else:
            alpha2_up = alpha2
    
    #鉴于误差的存在，当alpha2_up很靠近边界时，就让其取边界值。
    if alpha2_up < 1e-8:
        alpha2_up = 0.0
    elif alpha2_up > (model.C - 1e-8):
        alpha2_up = model.C
    
    #超过容差仍不能优化时，跳出循环
    if np.abs(alpha2_up - alpha2) < eps * (alpha2_up + alpha2 + eps):
        return 0, model

    #根据新的alpha2计算alpha1
    alpha1_up = alpha1 + s * (alpha2 - alpha2_up)
    
    #计算阈值b，都是根据论文里的公式
    b1 = E1 + y1 * (alpha1_up - alpha1) * k11 + y2 * (alpha2_up - alpha2) * k12 + model.b
    b2 = E2 + y1 * (alpha1_up - alpha1) * k12 + y2 * (alpha2_up - alpha2) * k22 + model.b
    #谁有效用谁，都在边界时b1和b2之间的值都可以用，当b1和b2都有效时，它们相等。
    if 0 < alpha1_up and alpha1_up < model.C:
        b_up = b1
    elif 0 < alpha2_up and alpha2_up < model.C:
        b_up = b2
    else:
        b_up = (b1 + b2) / 2.
    
    model.b = b_up
    
    #若训练模型为线性核函数
    if model.user_linear_optim:
        model.w = model.w + y1 * (alpha1_up - alpha1) * model.X[i1] + y2 * (alpha2_up - alpha2) * model.X[i2]
    
    #在alphas矩阵中更新alpha1和alpha2的值
    model.alphas[i1] = alpha1_up
    model.alphas[i2] = alpha2_up
    
    #更新误差矩阵
    model.errors[i1] = 0
    model.errors[i2] = 0
    
    #更新差值
    for i in range(model.m):
        if 0 < model.alphas[i] < model.C:
            model.errors[i] += y1 * (alpha1_up - alpha1) * model.kernel(model.X[i1], model.X[i]) + \
                y2 * (alpha2_up - alpha2) * model.kernel(model.X[i2], model.X[i]) + model.b - b_up
            
    return 1, model

#%%


def examine_examples(i2, model):
    '''
    
    '''
    y2 = model.y[i2]
    alpha2 = model.alphas[i2]
    E2 = get_error(model, i2)
    r2 = E2 * y2
    
    #确定alpha1
    if (r2 < - tol and alpha2 < model.C) or (r2 > tol and alpha2 > 0):
        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            #选择Ei矩阵中差值最大的进行优化
            #想要|E1-E2|最大，只需要在E2为正时选择最小的Ei作为E1，否则选择最大
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)
            
            #将选好的i1，i2放到油画布步骤里去优化
            step_result, model = take_step(i1, i2, model)
            
            if step_result:
                return 1, model
        
        #对于剩下的alpha，选择所有非0，非C得alphas值进行优化，随机选择起始点
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0], np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
        
        #alpha2确定的情况下，如何选择alpha1？循环所有（m-1）个alphas，随机选择起始点
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
    
    #先看最上面的if语句，如果if条件不满足，说明KKT条件已满足，找其他样本进行优化，执行下面的语句，推出
    return 0, model

def fit(model):
    '''
    
    '''
    numChanged = 0
    examineAll = 1
    
    #计数器，记录优化时的循环次数
    loopnum = 0
    loopnum1 = 0
    loopnum2 = 0
    
    #当numChanged=0且examineAll=0时退出循环
    #实际上是顺序地执行完所有的样本，也就是第一个if中的循环，并且else中的循环没有可优化的alpha
    #目标函数收敛了，在容差之内并且满足KKT条件，则循环退出，如果执行3000次循环仍未收敛，也退出
    #确定alpha2
    while (numChanged > 0) or (examineAll):
        numChanged = 0
        if loopnum == 3000:
            break
        loopnum = loopnum + 1
        
        if examineAll:
            print(loopnum + 1)
            #记录外循环次数
            loopnum1 = loopnum1 + 1
            #依次选择alpha2，alpha1
            count = 0
            for i in range(model.alphas.shape[0]):
                print(count)
                count += 1
                examine_result, model = examine_examples(i, model)
                numChanged += examine_result
        else:
            
            #上面if语句执行完成,进入内循环
            loopnum2 = loopnum2 + 1
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_examples(i, model)
                numChanged += examine_result
            
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
         
    print('loopnum : ', loopnum, ' loopnum1 : ', loopnum1, ' loopnum2 : ', loopnum2)
    return model
#%%    
'''----------------生成测试数据，训练样本---------------'''        
X_train, y = make_blobs(n_samples = 1000, centers = 2, n_features = 2, random_state = 2)
scalar = StandardScaler()   #数据预处理，使得经过处理的数据符合正标准正态分布
#训练样本异常大或小也会影响样本的正确分类，分散也会影响
X_train_scaled = scalar.fit_transform(X_train, y)
y[y == 0] = -1

#%%
'''--------------设置模型参数和初始化--------------'''
C = 20.0    #正则化参数
m = len(X_train_scaled)     #样本个数
initial_alphas = np.zeros(m)    #初始化乘子向量
initial_b = 0.0     #初始化阈值

#设置极限值
tol = 0.01  #误差界
eps = 0.01  #alpha界    

#模型初始化
model = SMOStruct(X_train_scaled, y, C, lambda x, y : gaussian_kernel(x, y, sigma = 0.5), initial_alphas, initial_b, np.zeros(m), user_linear_optim = False)

initial_error = decision_function(model.alphas, model.y, model.kernel, model.X, model.X, model.b) - model.y

model.errors = initial_error

print('Srarting to fit ...')

t1 = time.time()

output = fit(model)

t2 = time.time()
print('训练耗时: ', t2 - t1)
#%%
'''------------结果可视化----------'''
fig, ax = plt.subplots()
grid, ax = plot_decision_boundary(output, ax)
plt.show()

    





