from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import deepxde as dde
import pandas as pd  
from deepxde.backend import tf
            
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

#the specific heat ratio 
gamma = 1.4


#Set the neural network inputs and outputs, and PDE to be solved. The input of the neural network is x and the output is rho,v,t,P.
def pde(x, F):
    rho,v,t,p = F[:, 0:1], F[:, 1:2], F[:, 2:3],F[:, 3:4]
    
    m = rho*v*(1+2.2*(x-1.5)**2)
    Mv = (rho*v**2)*(1+2.2*(x-1.5)**2)
    PAV= p*v*(1+2.2*(x-1.5)**2)
    E = rho*v*(1+2.2*(x-1.5)**2)*(2.5*t+0.5*v**2)
    
    
    rho_x = dde.grad.jacobian(F, x, i=0, j=0)   
    v_x = dde.grad.jacobian(F, x, i=1, j=0)
    t_x = dde.grad.jacobian(F, x, i=2, j=0)
    p_x = dde.grad.jacobian(F, x, i=3, j=0)
    
    m_x = (v*rho_x+rho*v_x+rho*v*4.4*(x-1.5)/(1+2.2*(x-1.5)**2))*(1+2.2*(x-1.5)**2)
    mv_x = (rho*v*v_x*gamma+p_x)*(1+2.2*(x-1.5)**2)
    e_x = v*t_x*(1+2.2*(x-1.5)**2)*rho+(1+2.2*(x-1.5)**2)*p*(gamma-1)*(v_x+v*4.4*(x-1.5)/(1+2.2*(x-1.5)**2))


    return m_x, e_x,  mv_x, p-rho*t


def A(x):
    return (1+2.2*(x-1.5)**2)


#Set boundary. x∈(1.5,2.25)

def boundary1(x,on_boundary):
    return on_boundary and np.isclose(x[0],1.5)
def boundary2(x,on_boundary):
    return on_boundary and np.isclose(x[0],2.25)


domain = dde.geometry.Interval(1.5,2.25)

#Set the boundary conditions at the inlet. 
boundary_F1_1 = dde.icbc.DirichletBC(domain,lambda x:0.634,boundary1, component=0)

boundary_F2_1 = dde.icbc.DirichletBC(domain,lambda x:0.912,boundary1, component=1)


boundary_F3_1 = dde.icbc.DirichletBC(domain,lambda x:0.833,boundary1, component=2)

boundary_F4_1 = dde.icbc.DirichletBC(domain,lambda x:0.528,boundary1, component=3)



#Set the number of training points and their distribution.
data = dde.data.PDE(
    domain,
    pde,
    [boundary_F1_1, boundary_F2_1,boundary_F3_1, boundary_F4_1],
    num_domain=200,
    num_boundary=2,

    train_distribution='uniform'
)


#Set the size of the neural network, activation function, initialization method

net = dde.nn.FNN([1] + 3 * [30] + [4], "tanh", "Glorot uniform")


model = dde.Model(data, net)

#Set optimizer parameters, weight distribution

model.compile("adam", lr=1e-3)
model.train(epochs=2000)
model.compile("L-BFGS")


losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
