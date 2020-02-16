#Ex1: Logestic_Regression / classification / single neuron neural network model



import numpy as np


p=10
size_data=500
size_mini_batch=40
epoch=50
theta_=np.random.normal(size=(p+1,1))#choose theta_ in R^{p+1} (we need to generat the data)

def data_generator(theta_,p, size_data):

    X_delta=np.random.normal(size=(size_data, p))
    X=np.ones((size_data,p+1))
    X[:,1:]=X_delta
    Y=(np.sign(np.matmul(X, theta_))+1)/2
    return X, Y

X,Y=data_generator(theta_,p,size_data)

def mini_batch_creator(X,Y,size_data,size_mini_batch):
    permu_=np.random.permutation(size_data)
    X=X[permu_,:]
    Y=Y[permu_,:]
    X_mini_batch_list=[X[i: min(i+size_mini_batch, size_data),:] for i in range(0,size_data,size_mini_batch)]
    Y_mini_batch_list=[Y[i: min(i+size_mini_batch, size_data),:] for i in range(0,size_data,size_mini_batch)]
    return X_mini_batch_list, Y_mini_batch_list

def sigma(C):
    return 1/(1+np.exp(-C))

def grad_cost_SGD(X,Y,theta,lamda):
    return np.transpose(np.sum((sigma(np.matmul(X,theta))-Y)*X, 0 ) + lamda* np.transpose(theta))


def Error(X,Y,theta):
    return np.sum(np.linalg.norm((np.sign(np.matmul(X, theta))+1)/2 -Y))


def SGD(X,Y,grad_cost, eta,lamda, gamma, size_data,size_mini_batch, epoch):
    V=np.zeros((p+1,1))
    theta=np.random.normal(size=(p+1,1))
    for j in range(0,epoch):
        X_mini_list, Y_mini_list= mini_batch_creator(X,Y,size_data,size_mini_batch)
        for i in range(0,len(Y_mini_list)):
            V=gamma*V + eta*grad_cost(X_mini_list[i],Y_mini_list[i],theta,lamda)
            theta=theta-V
        print('Epoch=', j, '\t Error=', Error(X,Y,theta)*100/size_data, "%")
    return



SGD(X,Y,grad_cost_SGD, 1,2, 3, size_data,size_mini_batch, epoch)
