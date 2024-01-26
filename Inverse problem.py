from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import deepxde as dde
import pandas as pd  
from deepxde.backend import tf
            
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

gamma = dde.Variable(0.0)


#Set the specific heat ratio to the variable to be optimised with an initial value of 0.
df1 = pd.read_excel(r'C:\Users\log4_50.xlsx',sheet_name='p',usecols=[1, 1]) 
arr = df1.to_numpy()
ob_p = arr.reshape(50,1)
ob_x = np.linspace(0, 2.25,50).reshape(50, 1)

#Set the neural network inputs and outputs, and PDE to be solved. The input of the neural network is x and the output is rho,v,t,P.

def pde(x, F):
    rho,v,t,p = F[:, 0:1], F[:, 1:2], F[:, 2:3],F[:, 3:4]

    
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

#Set hard constrained boundary conditions.Fix the loss of pressure at the boundary to 0.
def modify_output(x,F):
    
    rho, v, t,p = F[:, 0:1], F[:, 1:2], F[:,2:3],F[:,3:4]
    
    p_new = x*(x-2.25)*p+(-0.084369*x+1)

    return  tf.concat((rho,v,t,p_new), axis=1)


#Set boundary. x∈(0,2.25)
def boundary1(x,on_boundary):
    return on_boundary and np.isclose(x[0],0)
def boundary2(x,on_boundary):
    return on_boundary and np.isclose(x[0],2.25)


domain = dde.geometry.Interval(0,2.25)

#Set the boundary conditions at the inlet and outlet.
boundary_F1_1 = dde.icbc.DirichletBC(domain,lambda x:1,boundary1, component=0)

boundary_F3_1 = dde.icbc.DirichletBC(domain,lambda x:1,boundary1, component=2)

boundary_F4_1 = dde.icbc.DirichletBC(domain,lambda x:1,boundary1, component=3)

boundary_F4_2 = dde.icbc.DirichletBC(domain,lambda x:0.81017,boundary2, component=3)

observe_p = dde.icbc.PointSetBC(ob_x, ob_p, component=3)

#Set the number of training points and their distribution.
data = dde.data.PDE(
    domain,
    pde,
    [boundary_F1_1, boundary_F3_1,boundary_F4_1, boundary_F4_2,observe_p ],
    num_domain=2000,
    num_boundary=2,
    train_distribution='uniform'
)

#Set the size of the neural network, activation function, initialization method
net = dde.nn.FNN([1] + 3 * [30] + [4], "tanh", "Glorot uniform")

net.apply_output_transform(modify_output)


#Save the variables that need to be optimized
variable = dde.callbacks.VariableValue(gamma, period=500,filename="variables.dat")
model = dde.Model(data, net)


#Set optimizer parameters, weight distribution
model.compile("adam", lr=0.0001,external_trainable_variables=gamma,loss_weights=[1, 1,20,1, 1, 1,1,1 ,1])
model.train(iterations=25000,callbacks=[variable],display_every=1)


model.compile("L-BFGS",external_trainable_variables=gamma)
model.train(callbacks=[variable],display_every=1000)
losshistory, train_state = model.train(callbacks=[variable],display_every=1000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

