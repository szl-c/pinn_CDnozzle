from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import deepxde as dde
import pandas as pd  
from deepxde.backend import tf 
df = pd.read_excel(r'C:\Users\1.875.xlsx')
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False


#Set the neural network inputs and outputs, and PDE to be solved. The input of the neural network is x,t and the output is rho,v,t,P.
def pde(x, F):
    rho,v,t,p = F[:, 0:1], F[:, 1:2], F[:, 2:3],F[:, 3:4]
    
    rho_t = dde.grad.jacobian(F, x, i=0, j=1)
    v_t = dde.grad.jacobian(F, x, i=1, j=1)
    t_t = dde.grad.jacobian(F, x, i=2, j=1)
    
    rho_x = dde.grad.jacobian(F, x, i=0, j=0)   
    v_x = dde.grad.jacobian(F, x, i=1, j=0)
    t_x = dde.grad.jacobian(F, x, i=2, j=0)
    p_x = dde.grad.jacobian(F, x, i=3, j=0)
    
    m_x = (v*rho_x+rho*v_x+rho*v*4.4*(x[:,0:1]-1.5)/(1+2.2*(x[:,0:1]-1.5)**2))*(1+2.2*(x[:,0:1]-1.5)**2)
    mv_x = (rho*v*v_x*1.4+p_x)*(1+2.2*(x[:,0:1]-1.5)**2)
    e_x = t_x*(1+2.2*(x[:,0:1]-1.5)**2)*rho*v*2.5+(1+2.2*(x[:,0:1]-1.5)**2)*p*(v_x+v*4.4*(x[:,0:1]-1.5)/(1+2.2*(x[:,0:1]-1.5)**2))

    m_t = (1+2.2*(x[:,0:1]-1.5)**2)*rho_t
    mv_t = (1+2.2*(x[:,0:1]-1.5)**2)*rho*v_t*1.4
    e_t = (1+2.2*(x[:,0:1]-1.5)**2)*rho*t_t*2.5


    return m_x+m_t, (e_x+e_t ),  (mv_x+mv_t),  p-rho*t

def A(x):
    return (1+2.2*(x-1.5)**2)

#Set hard constrained boundary conditions
def modify_output(x,F):
    
    rho, v, t,p = F[:, 0:1], F[:, 1:2], F[:,2:3],F[:,3:4]
    
    rho_new =x[:,0:1]*rho+1
    v_new = 1*v
    t_new =x[:,0:1]*t+1
    p_1 = x[:,0:1]*(x[:,0:1]-2.25)*p+(-0.084369*x[:,0:1]+1)
    return  tf.concat((rho,v,t,p_1), axis=1)


#Set boundary condition and Initial condition

def boundary1(x,on_boundary):
    return on_boundary and np.isclose(x[0],0)
def boundary2(x,on_boundary):
    return on_boundary and np.isclose(x[0],2.25)
def initial(x,on_initial):
    return on_initial and np.isclose(x[1],0)

geom = dde.geometry.Interval(0,2.25)
time = dde.geometry.TimeDomain(0, 15)
geomtime = dde.geometry.GeometryXTime(geom, time)


boundary_F1_1 = dde.icbc.DirichletBC(geomtime,lambda x:1,boundary1, component=0)

boundary_F3_1 = dde.icbc.DirichletBC(geomtime,lambda x:1,boundary1, component=2)

boundary_F4_1 = dde.icbc.DirichletBC(geomtime,lambda x:1,boundary1, component=3)


boundary_F4_2 = dde.icbc.DirichletBC(geomtime,lambda x: 0.81017 ,boundary2, component=3)


ic_rho = dde.icbc.IC(geomtime,lambda x: 1, initial, component=0)
ic_v = dde.icbc.IC(geomtime,lambda x: 0, initial, component=1)
ic_T = dde.icbc.IC(geomtime,lambda x:1, initial, component=2)
ic_p = dde.icbc.IC(geomtime,lambda x:1, initial, component=3)

#Add additional training points
j0, k0 = np.meshgrid(np.linspace(0, 2.25, 50), np.linspace(7.5,15, 50))
Anchors = np.vstack((np.ravel(j0), np.ravel(k0))).T


#Set the number of training points and their distribution
data = dde.data.TimePDE(
    geomtime,
    pde,
    [boundary_F1_1, boundary_F3_1,ic_p,ic_v,ic_rho,ic_T],
    num_domain=10000,
    num_boundary=50,
    num_initial=150,
    train_distribution='uniform'
)

#Set the size of the neural network, activation function, initialization method
net = dde.nn.FNN([2] + 4 * [50] + [4], "tanh", "Glorot uniform")

net.apply_output_transform(modify_output)


model = dde.Model(data, net)

model.compile("adam",lr=1e-4,loss_weights=[1, 1,20,1, 1, 1,1,1 ,1,1])
model.train(epochs=10000)
model.compile("L-BFGS")

losshistory, train_state = model.train()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)


