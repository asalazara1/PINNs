import tensorflow as tf
import numpy as np
import deepxde as dde
import time

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
pi = tf.constant(np.pi,dtype=DTYPE)

def residual(u,u_t,u_x,u_y,u_xx,u_yy,v,v_t,v_x,v_y,v_xx,v_yy,p_x,p_y, l1, x,y,t, source):
    
    f_val, g_val = source(x,y,t)

    f_u = u_t + u*u_x + v*u_y- l1*u_xx - l1*u_yy + p_x - f_val
    f_v = v_t + u*v_x + v*v_y- l1*v_xx - l1*v_yy + p_y - g_val
    return f_u, f_v

class PhysicsInformedNN:
    def __init__(self, lb, ub, layers, p0, xp0, psi0, x0, X, lambda1, source):

        self.lambda1 = lambda1
        self.lb = lb
        self.ub = ub
                
        self.layers = layers

        self.u0 = psi0
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
    

    def loss(self, X, X0, psi0, Xp0, p0):

        psi_p_pred = self.model(X0)
        psi_pred = psi_p_pred[:,0:1]
        
        u_p_pred2 = self.model(Xp0)
        p_pred = u_p_pred2[:,1:2]

        loss = tf.reduce_mean(tf.square(psi0-psi_pred))

        loss += tf.reduce_mean(tf.square(p0-p_pred))

        r1, r2 = self.get_residual(X)

        phi_ru = tf.reduce_mean(tf.square(r1))

        phi_rv = tf.reduce_mean(tf.square(r2))

        loss += phi_ru

        loss += phi_rv

        return loss
    
    def get_residual(self,X):
        with tf.GradientTape(persistent=True) as tape:
            x = X[:,0:1]
            y = X[:,1:2]
            t = X[:,2:3]

            tape.watch(x)
            tape.watch(y)
            tape.watch(t)

            psi_p = self.model(tf.stack([x[:,0],y[:,0],t[:,0]], axis=1))
            
            psi = psi_p[:,0:1]
            p = psi_p[:,1:2]

            u = tape.gradient(psi,y)
            v = -tape.gradient(psi,x)

            u_t = tape.gradient(u,t)
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            u_xx = tape.gradient(u_x,x)
            u_yy = tape.gradient(u_y,y)

            v_t = tape.gradient(v,t)
            v_x = tape.gradient(v,x)
            v_y = tape.gradient(v,y)
            v_xx = tape.gradient(v_x,x)
            v_yy = tape.gradient(v_y,y)

            p_x = tape.gradient(p,x)
            p_y = tape.gradient(p,y)


        del tape
        
        l1 = self.lambda1
        source = self.source
        f_u, f_v = residual(u,u_t,u_x,u_y,u_xx,u_yy,v,v_t,v_x,v_y,v_xx,v_yy,p_x,p_y, l1, x,y,t, source)

        return f_u, f_v
    

layers1 = [3, 50, 50, 50, 50, 50, 2]
layers2 = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
layers3 = [3, 50, 50, 50, 50, 50, 50, 50, 50, 2]
layers4 = [3, 70, 70, 70, 70, 70, 70, 70, 70, 2]
layers5 = [3, 100, 100, 100, 100, 100, 100, 100, 100, 2]

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
xmin = -1.
xmax = 1.
ymin = -1.
ymax = 1.

# Lower bounds
lb = tf.constant([xmin, ymin, tmin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([ xmax, ymax,tmax], dtype=DTYPE)


# Draw uniform sample points for initial boundary data
t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[2]
x_0 = tf.random.uniform((N_0,1), lb[0], ub[0], dtype=DTYPE)
y_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)
X_0 = tf.concat([x_0 ,y_0, t_0], axis=1)

# Boundary data
t_b = tf.random.uniform((N_b,1), lb[2], ub[2], dtype=DTYPE)

x_b1 = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((int(N_b/2),1), 0.5, dtype=DTYPE)

y_b1 = tf.random.uniform((int(N_b/2),1), lb[1], ub[1], dtype=DTYPE)

x_b2 = tf.random.uniform((int(N_b/2),1), lb[0], ub[0], dtype=DTYPE)

y_b2 = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((int(N_b/2),1), 0.5, dtype=DTYPE)

x_b = tf.concat([x_b1,x_b2], axis=0)

y_b = tf.concat([y_b1,y_b2], axis=0)

X_b = tf.concat([x_b, y_b,  t_b], axis=1)

# Boundary data for preassure
t_bp = tf.random.uniform((N_b,1), lb[2], ub[2], dtype=DTYPE)

x_bp1 = lb[0] + (ub[0] - lb[0]) * tf.keras.backend.random_bernoulli((int((N_b+N_0)/2),1), 0.5, dtype=DTYPE)

y_bp1 = tf.random.uniform((int((N_b+N_0)/2),1), lb[1], ub[1], dtype=DTYPE)

x_bp2 = tf.random.uniform((int((N_b+N_0)/2),1), lb[0], ub[0], dtype=DTYPE)

y_bp2 = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((int((N_b+N_0)/2),1), 0.5, dtype=DTYPE)

x_bp = tf.concat([x_b1,x_b2], axis=0)

y_bp = tf.concat([y_b1,y_b2], axis=0)

X_bp = tf.concat([x_bp, y_bp,  t_bp], axis=1)


# Draw uniformly sampled collocation points
t_r = tf.random.uniform((N_r,1), lb[2], ub[2], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
y_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
X_r = tf.concat([ x_r, y_r, t_r], axis=1)

# Collect boundary and inital data in lists
X_data = tf.concat([X_0, X_b],0)

########################   CASE 1   ########################

print('Caso 1')

def boundary_u1(t,x,y,lb,ub):
    I1 = tf.cast(tf.logical_or(x == lb[0] , x == ub[0]),DTYPE) 
    I2 = tf.cast(tf.logical_or(y == lb[1] , y == ub[1]),DTYPE)
    return I1 *( 1-tf.pow(y,2))*tf.exp(-t)  + I2*( 1-tf.pow(x,2))*tf.exp(-t) 

def boundary_p1(t,x,y):
    return x * y * tf.exp(-t) 

def initial_1(x,y):
    return 1-tf.pow(x,2)*tf.pow(y,2)


def source_1(x,y,t):
    f = 2*tf.pow(x,2)*y* tf.exp(-t) + 12* tf.pow(x,3) * tf.pow(y,2) * tf.exp(-t) + 4*y*tf.exp(-t) + y
    g = -2*tf.pow(y,2)*x* tf.exp(-t) + 4* tf.pow(x,2) * tf.pow(y,3) * tf.exp(-t) - 4*x*tf.exp(-t) + x
    return f,g

l1 = 1

# Evaluate intitial condition at x_0
u_0 = initial_1(x_0, y_0)

# Evaluate boundary condition at (t_b,x_b)
u_b = boundary_u1(t_b, x_b, y_b, lb,ub)

p_b = boundary_p1(t_bp, x_bp, y_bp)

# Collect boundary and inital data in lists
u_data = tf.concat([u_0, u_b],0)

# Setting up model 1
# model11 = PhysicsInformedNN(lb, ub, layers1, p_b, X_bp, u_data, X_data, X_r, l1, source_1)
# model12 = PhysicsInformedNN(lb, ub, layers2, p_b, X_bp, u_data, X_data, X_r, l1, source_1)
# model13 = PhysicsInformedNN(lb, ub, layers3, p_b, X_bp, u_data, X_data, X_r, l1, source_1)
model14 = PhysicsInformedNN(lb, ub, layers4, p_b, X_bp, u_data, X_data, X_r, l1, source_1)
# model15 = PhysicsInformedNN(lb, ub, layers5, p_b, X_bp, u_data, X_data, X_r, l1, source_1)

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

print('Arquitectura 4')

def time_step14():
        loss = model14.loss(model14.X, model14.X0, model14.u0, model14.Xp0, model14.p0)
        return loss

variables14 = model14.model.trainable_variables

dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

start = time.time()

dde.optimizers.tfp_optimizer.lbfgs_minimize(variables14, time_step14)

stop = time.time()

duration = stop-start

print("End trainign, time:",duration)

model14.model.export('PINN_Burgers_C1_L4_Export')

# print('Arquitectura 5')

# def time_step15():
#         loss = model15.loss(model15.X, model15.X0, model15.u0, model15.Xp0, model15.p0)
#         return loss

# variables15 = model15.model.trainable_variables

# dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

# start = time.time()

# dde.optimizers.tfp_optimizer.lbfgs_minimize(variables15, time_step15)

# stop = time.time()

# duration = stop-start

# print("End trainign, time:",duration)

# model15.model.export('PINN_Burgers_C1_L5_Export')

