import tensorflow as tf
import numpy as np
import deepxde as dde
import time

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
pi = tf.constant(np.pi,dtype=DTYPE)

def residual(u,u_t,u_x,u_xx, l1, x,t, source):
    
    f_val = source(x,t)
    f_u = u_t + u*u_x- l1*u_xx - f_val

    return f_u

class PhysicsInformedNN:
    def __init__(self, lb, ub, layers, u0, x0, X, lambda1, source):

        self.lambda1 = lambda1
        self.lb = lb
        self.ub = ub
                
        self.layers = layers

        self.u0 = u0
        self.X0 = x0

        self.X = X

        self.model = self.initialize_NN(layers)

        self.source = source

    def initialize_NN(self,layers):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(layers[0]))
        scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0*(x-self.lb)/(self.ub-self.lb)-1.0)
        model.add(scaling_layer)
        num_layers = len(layers)
        for i in range(1,num_layers-1):
            model.add(tf.keras.layers.Dense(layers[i],
                                            activation=tf.keras.activations.get('gelu'),
                                            kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(layers[-1],
                                            kernel_initializer='glorot_normal'))

        return model
    

    def loss(self, X, X0, u0):
        u_pred = self.model(X0)
        
        loss = tf.reduce_mean(tf.square(u0-u_pred))

        r1 = self.get_residual(X)

        phi_ru = tf.reduce_mean(tf.square(r1))

        loss += phi_ru

        return loss
    
    def get_residual(self,X):
        with tf.GradientTape(persistent=True) as tape:
            x = X[:,0:1]
            t = X[:,1:2]

            tape.watch(x)
            tape.watch(t)

            u = self.model(tf.stack([x[:,0],t[:,0]], axis=1))


            u_t = tape.gradient(u,t)
            u_x = tape.gradient(u,x)
            u_xx = tape.gradient(u_x,x)


        del tape
        
        l1 = self.lambda1
        source = self.source
        f_u = residual(u,u_t,u_x,u_xx,l1, x, t, source)

        return f_u
    

layers = [2, 20, 20, 20, 20, 20, 1]
layers2 = [2, 50, 50, 50, 50, 50, 1]
layers3 = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

cor=50
tol=1.0  * np.finfo(float).eps
iter=50000
fun=50000
ls=50

N_0 = 50
N_b = 50
N_r = 10000

tmin = 0.
tmax = 1.
xmin = 0.
xmax = 1.

# Lower bounds
lb = tf.constant([ xmin, tmin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([xmax, tmax], dtype=DTYPE)


# Draw uniform sample points for initial boundary data
t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[1]
x_0 = tf.random.uniform((N_0,1), lb[0], ub[0], dtype=DTYPE)
X_0 = tf.concat([x_0 , t_0], axis=1)

# Boundary data
t_b = tf.random.uniform((N_b,1), lb[1], ub[1], dtype=DTYPE)
x_b = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
X_b = tf.concat([x_b, t_b], axis=1)

# Draw uniformly sampled collocation points
t_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
X_r = tf.concat([ x_r,t_r], axis=1)

# Collect boundary and inital data in lists
X_data = tf.concat([X_0, X_b],0)

# ########################   CASE 1   ########################

# print('Caso 1')

# def boundary_1(t,x):
#     n = x.shape[0]
#     return tf.zeros((n,1),dtype = DTYPE)

# def initial_1(x):
#     return tf.sin(pi*x)

# def source_1(x,t):
#     return pi*tf.exp(-2*t)*tf.sin(pi*x)*tf.cos(pi*x)

# l1 = 1/tf.constant(np.power(np.pi,2),dtype='float32')

# # Evaluate intitial condition at x_0
# u_0 = initial_1(x_0)

# # Evaluate boundary condition at (t_b,x_b)
# u_b = boundary_1(t_b, x_b)

# # Collect boundary and inital data in lists
# u_data = tf.concat([u_0, u_b],0)

# # Setting up model 1
# model11 = PhysicsInformedNN(lb, ub, layers, u_data, X_data, X_r, l1, source_1)
# model12 = PhysicsInformedNN(lb, ub, layers2, u_data, X_data, X_r, l1, source_1)
# model13 = PhysicsInformedNN(lb, ub, layers3, u_data, X_data, X_r, l1, source_1)

# print('Arquitectura 1')

# def time_step11():
#         loss = model11.loss(model11.X, model11.X0, model11.u0)
#         return loss

# variables11 = model11.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables11, time_step11)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model11.model.export('PINN_Burgers_C1_L1_Export')

# print('Arquitectura 2')

# def time_step12():
#         loss = model12.loss(model12.X, model12.X0, model12.u0)
#         return loss

# variables12 = model12.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables12, time_step12)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model12.model.export('PINN_Burgers_C1_L2_Export')

# print('Arquitectura 3')

# def time_step13():
#         loss = model13.loss(model13.X, model13.X0, model13.u0)
#         return loss

# variables13 = model13.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables13, time_step13)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model13.model.export('PINN_Burgers_C1_L3_Export')

########################   CASE 2   ########################

def boundary_2(t,x):
    n = x.shape[0]
    return tf.zeros((n,1),dtype = DTYPE)

def initial_2(x):
    return x-tf.pow(x,2)

def source_2(x,t):
    return tf.exp(t)*(x-tf.pow(x,2)) + tf.exp(2*t)*(x-tf.pow(x,2))*(1-2*x) + 2*tf.exp(t)

l2 = 1

layers2 = [2, 20, 20, 20, 20, 20, 1]
# Evaluate intitial condition at x_0
u_02 = initial_2(x_0)

# Evaluate boundary condition at (t_b,x_b)
u_b2 = boundary_2(t_b, x_b)

# Collect boundary and inital data in lists
u_data2 = tf.concat([u_02, u_b2],0)

# Setting up model 1
# model21 = PhysicsInformedNN(lb, ub, layers, u_data2, X_data, X_r, l2, source_2)
model22 = PhysicsInformedNN(lb, ub, layers2, u_data2, X_data, X_r, l2, source_2)
# model23 = PhysicsInformedNN(lb, ub, layers3, u_data2, X_data, X_r, l2, source_2)

# print('Arquitectura 1')

# def time_step21():
#         loss = model21.loss(model21.X, model21.X0, model21.u0)
#         return loss

# variables21 = model21.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables21, time_step21)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model21.model.export('PINN_Burgers_C2_L1_Export')

print('Arquitectura 2')

def time_step22():
        loss = model22.loss(model22.X, model22.X0, model22.u0)
        return loss

variables22 = model22.model.trainable_variables

dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

start = time.time()

dde.optimizers.tfp_optimizer.lbfgs_minimize(variables22, time_step22)

stop = time.time()

duration = stop-start

print("End trainign, time:",duration)

model22.model.export('PINN_Burgers_C2_L2_Export')

# print('Arquitectura 3')

# def time_step23():
#         loss = model23.loss(model23.X, model23.X0, model23.u0)
#         return loss

# variables23 = model23.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables23, time_step23)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model23.model.export('PINN_Burgers_C2_L3_Export')

########################   CASE 3   ########################

def boundary_3(t,x):
    return t

def initial_3(x):
    n = x.shape[0]
    return tf.zeros((n,1),dtype = DTYPE)

def source_3(x,t):
    return tf.exp(x-tf.pow(x,2)) + tf.pow(t,2)*tf.exp(2*x-2*tf.pow(x,2))*(1-2*x) - t*tf.exp(x-tf.pow(x,2))*(4*tf.pow(x,2)-4*x-1)

l3 = 1

layers3 = [2, 20, 20, 20, 20, 20, 1]
# Evaluate intitial condition at x_0
u_03 = initial_3(x_0)

# Evaluate boundary condition at (t_b,x_b)
u_b3 = boundary_3(t_b, x_b)

# Collect boundary and inital data in lists
u_data3 = tf.concat([u_03, u_b3],0)

# Setting up model 1
model31 = PhysicsInformedNN(lb, ub, layers, u_data3, X_data, X_r, l3, source_3)
model32 = PhysicsInformedNN(lb, ub, layers2, u_data3, X_data, X_r, l3, source_3)
model33 = PhysicsInformedNN(lb, ub, layers3, u_data3, X_data, X_r, l3, source_3)

print('Arquitectura 1')

def time_step31():
        loss = model31.loss(model31.X, model31.X0, model31.u0)
        return loss

variables31 = model31.model.trainable_variables

dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

start = time.time()

dde.optimizers.tfp_optimizer.lbfgs_minimize(variables31, time_step31)

stop = time.time()

duration = stop-start

print("End trainign, time:",duration)

model31.model.export('PINN_Burgers_C3_L1_Export')

print('Arquitectura 2')

def time_step32():
        loss = model32.loss(model32.X, model32.X0, model32.u0)
        return loss

variables32 = model32.model.trainable_variables

dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

start = time.time()

dde.optimizers.tfp_optimizer.lbfgs_minimize(variables32, time_step32)

stop = time.time()

duration = stop-start

print("End trainign, time:",duration)

model32.model.export('PINN_Burgers_C3_L2_Export')

print('Arquitectura 3')

def time_step33():
        loss = model33.loss(model33.X, model33.X0, model33.u0)
        return loss

variables33 = model33.model.trainable_variables

dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

start = time.time()

dde.optimizers.tfp_optimizer.lbfgs_minimize(variables33, time_step33)

stop = time.time()

duration = stop-start

print("End trainign, time:",duration)

model33.model.export('PINN_Burgers_C3_L3_Export')