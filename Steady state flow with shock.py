#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import deepxde as dde
import pandas as pd  
from deepxde.backend import tf
df = pd.read_excel(r'C:\Users\1.875.xlsx')            
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

gamma = 1.4
R = 287.0


#Set the neural network inputs and outputs ,and the PDE
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

#Set hard constrained boundary conditions
def modify_output(x,F):
    
    rho, v, t,p = F[:, 0:1], F[:, 1:2], F[:,2:3],F[:,3:4]
    
    rho_new =x[:,0:1]*rho+1
    v_new = 1*v
    t_new = x[:,0:1]*t+1
    p_new = x*(x-2.25)*p+(-0.084369*x+1)

    return  tf.concat((rho,v_new,t,p_new), axis=1)


#Set boundary condition
def boundary1(x,on_boundary):
    return on_boundary and np.isclose(x[0],0)
def boundary2(x,on_boundary):
    return on_boundary and np.isclose(x[0],2.25)


domain = dde.geometry.Interval(0,2.25)


boundary_F1_1 = dde.icbc.DirichletBC(domain,lambda x:1,boundary1, component=0)


boundary_F3_1 = dde.icbc.DirichletBC(domain,lambda x:1,boundary1, component=2)
boundary_F4_1 = dde.icbc.DirichletBC(domain,lambda x:1,boundary1, component=3)


boundary_F4_2 = dde.icbc.DirichletBC(domain,lambda x:0.81017,boundary2, component=3)

#Set the number of training points and their distribution
data = dde.data.PDE(
    domain,
    pde,
    [boundary_F1_1, boundary_F3_1,boundary_F4_1, boundary_F4_2],
    num_domain=1000,
    num_boundary=2,

    train_distribution='uniform'
)


#Set the size of the neural network, activation function, initialization method

net = dde.nn.FNN([1] + 3 * [30] + [4], "tanh", "Glorot uniform")
net.apply_output_transform(modify_output)

model = dde.Model(data, net)

#Set optimizer parameters, weight distribution

model.compile("adam", lr=1e-3,loss_weights=[1, 1, 20,1, 1, 1,1,1 ])
model.train(epochs=10000)
model.compile("L-BFGS")


losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


#Output result
x=np.linspace(0,2.25,50).reshape(50,1)
F = model.predict(x)

F1 = F[:, 0]
F2 = abs(F[:, 1])
F3 = F[:, 2]
F4 = F[:, 3]
ma=F2/np.sqrt(F3)

data1 = pd.DataFrame(F1.reshape(50,1))
data2 = pd.DataFrame(F2.reshape(50,1))
data3 = pd.DataFrame(F3.reshape(50,1))
data4 = pd.DataFrame(F4.reshape(50,1))
data5 = pd.DataFrame(ma.reshape(50,1))
writer = pd.ExcelWriter('data.xlsx')
data1.to_excel(writer, 'rho', float_format='%.5f')
data2.to_excel(writer, 'v', float_format='%.5f')
data3.to_excel(writer, 't', float_format='%.5f')
data4.to_excel(writer, 'p', float_format='%.5f')
data5.to_excel(writer, 'ma', float_format='%.5f')
writer.save()
writer.close()


plt.figure(num=1)

plt.plot(x, F1, color="red", linewidth=1.0, linestyle="-")

plt.xlabel(' x ')
plt.ylabel(' rho/rho0 ')
plt.title('Density')

plt.figure(num=2)

plt.plot(x, F3, color="red", linewidth=1.0, linestyle="-")

plt.xlabel(' x ')
plt.ylabel(' T/T0 ')
plt.title('Temperature')

plt.figure(num=3)
x0 = df['x']
p0 = df['p']
plt.plot(x, F4, color="red", linewidth=1.0, linestyle="-")
plt.plot(x0, p0, color="blue", linewidth=1.0, linestyle="-")
plt.xlabel(' x ')
plt.ylabel(' p/p0 ')
plt.title('Presssure')


plt.figure(num=4)

plt.plot(x, ma, color="red", linewidth=1.0, linestyle="-")

plt.xlabel(' x ')
plt.ylabel(' Ma ')
plt.title('Mach number')
plt.show()



a=np.linspace(0,2.25,50).reshape(50,1)
K=model.predict(a)
k1=K[:,0]
k2=K[:,1]
k3=K[:,2]
k4=K[:,3]
print(k1)
print(k2)
print(k3)
print(k4)


# In[ ]:





# In[ ]:





# In[ ]:




