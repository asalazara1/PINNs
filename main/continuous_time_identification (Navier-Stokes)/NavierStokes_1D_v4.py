import tensorflow as tf
import numpy as np
import deepxde as dde
import time

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
pi = tf.constant(np.pi,dtype=DTYPE)

def residual(u,u_t,u_x,u_xx,p_x, l1, x,t, source):
    
    f_val = source(x,t)
    f_u = u_t + u*u_x- l1*u_xx + p_x - f_val

    return f_u

class PhysicsInformedNN:
    def __init__(self, lb, ub, layers, p0, xp0, u0, x0, X, lambda1, source):

        self.lambda1 = lambda1
        self.lb = lb
        self.ub = ub
                
        self.layers = layers

        self.u0 = u0
        self.X0 = x0

        self.p0 = p0
        self.Xp0 = xp0

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
    

    def loss(self, X, X0, u0, Xp0, p0):

        u_p_pred = self.model(X0)
        u_pred = u_p_pred[:,0:1]
        
        u_p_pred2 = self.model(Xp0)
        p_pred = u_p_pred2[:,1:2]

        loss = tf.reduce_mean(tf.square(u0-u_pred))

        loss += tf.reduce_mean(tf.square(p0-p_pred))

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

            u_p = self.model(tf.stack([x[:,0],t[:,0]], axis=1))
            
            u = u_p[:,0:1]
            p = u_p[:,1:2]


            u_t = tape.gradient(u,t)
            u_x = tape.gradient(u,x)
            u_xx = tape.gradient(u_x,x)

            p_x = tape.gradient(p,x)


        del tape
        
        l1 = self.lambda1
        source = self.source
        f_u = residual(u,u_t,u_x,u_xx,p_x, l1, x,t, source)

        return f_u
    

layers = [2, 20, 20, 20, 20, 20, 2]
layers2 = [2, 50, 50, 50, 50, 50, 2]
layers3 = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2]
layers4 = [2, 50, 50, 50, 50, 50, 50, 50, 50, 2]

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
lb = tf.constant([xmin, tmin], dtype=DTYPE)
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

# Boundary data for preassure
t_bp = tf.random.uniform((N_b + N_0,1), lb[1], ub[1], dtype=DTYPE)
x_bp = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((N_b + N_0,1), 0.5, dtype=DTYPE)
X_bp = tf.concat([x_bp, t_bp], axis=1)

# Draw uniformly sampled collocation points
t_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
X_r = tf.concat([ x_r,t_r], axis=1)

# Collect boundary and inital data in lists
X_data = tf.concat([X_0, X_b],0)

########################   CASE 1   ########################

# print('Caso 1')

# def boundary_u1(t,x):
#     n = x.shape[0]
#     return tf.zeros((n,1),dtype = DTYPE)

# def boundary_p1(t,x):
#     n = x.shape[0]
#     return tf.zeros((n,1),dtype = DTYPE)

# def initial_1(x):
#     return tf.sin(pi*x)


# def source_1(x,t):
#     return pi*tf.exp(-2*t)*tf.sin(pi*x)*tf.cos(pi*x) - pi*tf.cos(pi*x)

# l1 = 1/tf.constant(np.power(np.pi,2),dtype='float32')

# # Evaluate intitial condition at x_0
# u_0 = initial_1(x_0)

# # Evaluate boundary condition at (t_b,x_b)
# u_b = boundary_u1(t_b, x_b)

# p_b = boundary_p1(t_bp, x_bp)

# # Collect boundary and inital data in lists
# u_data = tf.concat([u_0, u_b],0)

# # Setting up model 1
# model11 = PhysicsInformedNN(lb, ub, layers, p_b, X_bp, u_data, X_data, X_r, l1, source_1)
# model12 = PhysicsInformedNN(lb, ub, layers2, p_b, X_bp, u_data, X_data, X_r, l1, source_1)
# model13 = PhysicsInformedNN(lb, ub, layers3, p_b, X_bp, u_data, X_data, X_r, l1, source_1)
# model14 = PhysicsInformedNN(lb, ub, layers4, p_b, X_bp, u_data, X_data, X_r, l1, source_1)

# print('Arquitectura 1')

# def time_step11():
#         loss = model11.loss(model11.X, model11.X0, model11.u0, model11.Xp0, model11.p0)
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
#         loss = model12.loss(model12.X, model12.X0, model12.u0, model12.Xp0, model12.p0)
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
#         loss = model13.loss(model13.X, model13.X0, model13.u0, model13.Xp0, model13.p0)
#         return loss

# variables13 = model13.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables13, time_step13)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model13.model.export('PINN_Burgers_C1_L3_Export')

# print('Arquitectura 4')

# def time_step14():
#         loss = model14.loss(model14.X, model14.X0, model14.u0, model14.Xp0, model14.p0)
#         return loss

# variables14 = model14.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables14, time_step14)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model14.model.export('PINN_Burgers_C1_L4_Export')

########################   CASE 2   ########################

print('Caso 2')

def boundary_u2(t,x):
    n = x.shape[0]
    return tf.zeros((n,1),dtype = DTYPE)

def boundary_p2(t,x):
    n = x.shape[0]
    return tf.zeros((n,1),dtype = DTYPE)

def initial_2(x):
    return x-tf.pow(x,2)

def source_2(x,t):
    return tf.exp(t)*(x-tf.pow(x,2)) + tf.exp(2*t)*(x-tf.pow(x,2))*(1-2*x) + 2*tf.exp(t) - pi*tf.cos(pi*x)

l2 = 1
# Evaluate intitial condition at x_0
u_02 = initial_2(x_0)

# Evaluate boundary condition at (t_b,x_b)
u_b2 = boundary_u2(t_b, x_b)

p_b2 = boundary_p2(t_bp, x_bp)
# Collect boundary and inital data in lists
u_data2 = tf.concat([u_02, u_b2],0)

# Setting up model 1
model21 = PhysicsInformedNN(lb, ub, layers, p_b2, X_bp, u_data2, X_data, X_r, l2, source_2)
# model22 = PhysicsInformedNN(lb, ub, layers2, p_b2, X_bp, u_data2, X_data, X_r, l2, source_2)
# model23 = PhysicsInformedNN(lb, ub, layers3, p_b2, X_bp, u_data2, X_data, X_r, l2, source_2)
# model24 = PhysicsInformedNN(lb, ub, layers4, p_b2, X_bp, u_data2, X_data, X_r, l2, source_2)

print('Arquitectura 1')

def time_step21():
        loss = model21.loss(model21.X, model21.X0, model21.u0, model21.Xp0, model21.p0)
        return loss

variables21 = model21.model.trainable_variables

dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

start = time.time()

dde.optimizers.tfp_optimizer.lbfgs_minimize(variables21, time_step21)

stop = time.time()

duration = stop-start

print("End trainign, time:",duration)

model21.model.export('PINN_Burgers_C2_L1_Export')

# print('Arquitectura 2')

# def time_step22():
#         loss = model22.loss(model22.X, model22.X0, model22.u0, model22.Xp0, model22.p0)
#         return loss

# variables22 = model22.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables22, time_step22)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model22.model.export('PINN_Burgers_C2_L2_Export')

# print('Arquitectura 3')

# def time_step23():
#         loss = model23.loss(model23.X, model23.X0, model23.u0, model23.Xp0, model23.p0)
#         return loss

# variables23 = model23.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables23, time_step23)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model23.model.export('PINN_Burgers_C2_L3_Export')

# print('Arquitectura 4')

# def time_step24():
#         loss = model24.loss(model24.X, model24.X0, model24.u0, model24.Xp0, model24.p0)
#         return loss

# variables24 = model24.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables24, time_step24)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model24.model.export('PINN_Burgers_C2_L4_Export')
