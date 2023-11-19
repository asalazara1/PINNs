import tensorflow as tf
import numpy as np
import deepxde as dde
import time

# Set the data type 
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

pi = tf.constant(np.pi,dtype=DTYPE)

def fun_u_0(x):
    return tf.sin( pi * x)

def fun_p_b(t,x):
    n = x.shape[0]
    return tf.zeros((n,1),dtype = DTYPE)

def fun_u_b(t,x):
    n = x.shape[0]
    return tf.zeros((n,1),dtype = DTYPE)

def f(x,t):
    return pi*(tf.exp(-2*t)*tf.sin(pi*x)*tf.cos(pi*x) - tf.cos(pi*x))  

def residual(u,u_t,u_x,u_xx,p_x, l1, x,t):
    
    f_val = f(x,t)
    f_u = u_t + u*u_x- l1*u_xx + p_x - f_val

    return f_u

class PhysicsInformedNN:
    def __init__(self, lb, ub, layers, p0, xp0, u0, x0, X):

        self.lambda1 = 1/tf.constant(np.power(np.pi,2),dtype='float32')
        self.lb = lb
        self.ub = ub
                
        self.layers = layers

        self.u0 = u0
        self.X0 = x0

        self.p0 = p0
        self.Xp0 = xp0

        self.X = X

        self.model = self.initialize_NN(layers)

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
        loss = 2*tf.reduce_mean(tf.square(u0-u_pred))
        loss = loss + 2*tf.reduce_mean(tf.square(p0-p_pred))
        r1 = self.get_residual(X)

        phi_ru = tf.reduce_mean(tf.square(r1))

        loss = loss + phi_ru

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
        f_u = residual(u,u_t,u_x,u_xx,p_x,l1, x, t)

        return f_u

print("Creating Params")

layers = [2, 20, 20, 20, 2]

N_0 = 50
N_b = 100
N_r = 5000

tmin = 0.
tmax = 1.
xmin = -1.
xmax = 1.

# Lower bounds
lb = tf.constant([xmin, tmin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([xmax, tmax], dtype=DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(0)

# Draw uniform sample points for initial boundary data
t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[1]
x_0 = tf.random.uniform((N_0,1), lb[0], ub[0], dtype=DTYPE)
X_0 = tf.concat([x_0 , t_0], axis=1)

# Evaluate intitial condition at x_0
u_0 = fun_u_0(x_0)

# Boundary data
t_b = tf.random.uniform((N_b,1), lb[1], ub[1], dtype=DTYPE)
x_b = lb[1] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
X_b = tf.concat([x_b, t_b], axis=1)

# Evaluate boundary condition at (t_b,x_b)
u_b = fun_u_b(t_b, x_b)
p_b = fun_p_b(t_b, x_b)
# Draw uniformly sampled collocation points
t_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
X_r = tf.concat([x_r,t_r], axis=1)

# Collect boundary and inital data in lists
X_data = tf.concat([X_0, X_b],0)
u_data = tf.concat([u_0, u_b],0)

print("Setting up model")

model = PhysicsInformedNN(xmin, xmax, layers,  p_b, X_b, u_data, X_data, X_r)

def time_step():
        loss = model.loss(model.X, model.X0, model.u0, model.Xp0, model.p0)
        return loss

variables = model.model.trainable_variables

cor=50
tol=1.0  * np.finfo(float).eps
iter=50000
fun=50000
ls=50

dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

print("Optimizing model")

start = time.time()

dde.optimizers.tfp_optimizer.lbfgs_minimize(variables, time_step)

stop = time.time()

duration = stop-start

print("End trainign, time:",duration)

print("Exporting initial model")

model.model.export('PINN_1D_Export8')