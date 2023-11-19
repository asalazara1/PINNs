import tensorflow as tf
import numpy as np
import deepxde as dde
import time

dde.backend.set_default_backend("tensorflow")

DTYPE = 'float32'
pi = tf.constant(np.pi,dtype='float32')

def psi_real(x,y,t):
    return -1/(2*pi)*tf.sin(pi*(tf.pow(x,2)+tf.pow(y,2)))*tf.sin(tf.pow(t,2)+1)

def p_real(x,y,t):
    return tf.sin(x-y+t)



def residual(u,v,u_t,v_t,u_x,u_xx,u_y,u_yy,v_x,v_xx,v_y,v_yy,p_x,p_y,l1, l2, x, y, t):
    
    g_x, g_y = source(x,y,t,l1,l2) 

    f_u = u_t + l1*(u*u_x + v*u_y) + p_x - l2* (u_xx + u_yy) - g_x
    f_v = v_t + l1*(u*v_x + v*v_y) + p_y - l2* (v_xx + v_yy) - g_y

    return f_u, f_v

def source(x,y,t,l1,l2):

    
    r = tf.pow(x,2) + tf.pow(y,2)
    T = tf.pow(t,2) + 1
    g_x = -2*t*y*tf.sin(pi*r)*tf.cos(T) - l1*x*tf.pow(tf.sin(pi*r)*tf.sin(T),2) - 4 * l2* tf.pow(pi,2)*y*r*tf.sin(pi*r)*tf.sin(T) + 8* l2 * pi*y*tf.cos(pi*r)*tf.sin(T) + tf.cos(x-y+t)
    g_y =  2*t*x*tf.sin(pi*r)*tf.cos(T) - l1*y*tf.pow(tf.sin(pi*r)*tf.sin(T),2) + 4 * l2* tf.pow(pi,2)*x*r*tf.sin(pi*r)*tf.sin(T) - 8* l2 * pi*x*tf.cos(pi*r)*tf.sin(T) - tf.cos(x-y+t)

    return g_x, g_y

def fun_u_0(x, y):
    r = tf.pow(x,2) + tf.pow(y,2)
    return tf.constant(-np.sin(1)/(2*np.pi),dtype='float32') * tf.sin(pi*r) 

def fun_u_b(t,x, y):
    n = x.shape[0]
    return tf.zeros((n,1),dtype = DTYPE)


class PhysicsInformedNN:
    def __init__(self, lb, ub, layers, psi0, x0, X):

        self.lb = lb
        self.ub = ub
                
        self.layers = layers

        self.u0 = psi0
        self.X0 = x0
        self.X = X

        self.model = self.initialize_NN(layers)
        self.lambda1 = 1
        self.lambda2 = 1
    def initialize_NN(self,layers):
        model = tf.keras.Sequential()
        scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0*(x-self.lb)/(self.ub-self.lb)-1.0)
        
        model.add(tf.keras.Input(layers[0]))
        model.add(scaling_layer)

        num_layers = len(layers)
        for i in range(1,num_layers-1):
            model.add(tf.keras.layers.Dense(layers[i],
                                            activation=tf.keras.activations.get('tanh'),
                                            kernel_initializer='glorot_normal',
                                            bias_initializer='zeros'))
            
        model.add(tf.keras.layers.Dense(layers[-1]))

        return model
    

    def loss(self, X, X0, psi0,):
        psi_and_p = self.model(X0)
        psi_pred =  psi_and_p[:,0:1]
        p_pred = psi_and_p[:,1:2]

        loss = tf.reduce_mean(tf.square(psi0-psi_pred))

        r1,r2 = self.get_residual(X)

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

            psi_and_p = self.model(tf.stack([x[:,0],y[:,0],t[:,0]], axis=1))

            psi = psi_and_p[:,0:1]
            p = psi_and_p[:,1:2]

            u = tape.gradient(psi,y)
            v = -tape.gradient(psi,x) 

            u_t = tape.gradient(u,t)
            v_t = tape.gradient(v,t)

            u_x = tape.gradient(u,x)
            u_xx = tape.gradient(u_x,x)

            u_y = tape.gradient(u,y)
            u_yy = tape.gradient(u_y,y)

            v_x = tape.gradient(v,x)
            v_xx = tape.gradient(v_x,x)

            v_y = tape.gradient(v,y)
            v_yy = tape.gradient(v_y,y)

            p_x = tape.gradient(p,x)
            p_y = tape.gradient(p,y)

        del tape
        
        l1 = self.lambda1
        l2 = self.lambda2
        f_u, f_v = residual(u,v,u_t,v_t,u_x,u_xx,u_y,u_yy,v_x,v_xx,v_y,v_yy,p_x,p_y,l1, l2, x, y, t)

        return f_u, f_v
    
    def predict(self, X):
        with tf.GradientTape(persistent=True) as tape:
            x = X[:,0:1]
            y = X[:,1:2]
            t = X[:,2:3]

            tape.watch(x)
            tape.watch(y)
            tape.watch(t)

            psi_and_p = self.model(tf.stack([x[:,0],y[:,0],t[:,0]], axis=1))

            psi = psi_and_p[:,0:1]
            p = psi_and_p[:,1:2]

            u = tape.gradient(psi,y)
            v = -tape.gradient(psi,x) 
        del tape
        return u, v ,p 
    
    def loss_gradient(self,X,X0,u0, v0):

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss(self, X, X0, u0, v0)
            g = tape.gradient(loss, self.model.trainable_variables)

        del tape
        return loss, g

    def train(self):
        def time_step():
            loss = self.loss(self.X, self.X0, self.psi0)
            return loss
        variables = self.model.trainable_variables
        cor=50
        tol=1.0  * np.finfo(float).eps
        iter=50000
        fun=50000
        ls=50
        dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)
        dde.optimizers.tfp_optimizer.lbfgs_minimize(variables, time_step)

print("Creating Params")
layers1 = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
layers2 = [3, 50, 50, 50, 50, 50, 50, 50, 50, 2]
layers3 = [3, 100, 100, 100, 100, 100, 100, 100, 100, 2]

N_0 = 100
N_b = 100
N_r = 10000

tmin = 0.
tmax = 2.
xmin = -1.
xmax = 1.
ymin = -1
ymax = 1
# Lower bounds
lb = tf.constant([xmin, ymin, tmin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([xmax, ymax, tmax], dtype=DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(0)

# Draw uniform sample points for initial boundary data
t_b = tf.random.uniform((N_b,1), lb[2], ub[2], dtype=DTYPE)
x_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
y_b = np.zeros((N_b,1))
for i in range(x_b.shape[0]):
    y_b[i] = np.power(-1,i)*np.sqrt(1-np.power(x_b[i],2))

y_b = tf.convert_to_tensor(y_b, dtype=DTYPE)

X_b = tf.concat([x_b, y_b , t_b], axis=1)

t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[2]
x_0 = tf.random.uniform((N_0,1), lb[0], ub[0], dtype=DTYPE)
y_0 = np.zeros((N_0,1))
y_0 = tf.random.uniform((N_0,1),-1,1, dtype=DTYPE)*tf.sqrt(1-tf.pow(x_0,2))

X_0 = tf.concat([x_0, y_0 , t_0], axis=1)

u_0 = fun_u_0(x_0,y_0)

u_b = fun_u_b(t_b,x_b,y_b)

X_data = tf.concat([X_0, X_b],0)
u_data = tf.concat([u_0, u_b],0)

t_r = tf.random.uniform((N_r,1), lb[2], ub[2], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
y_r = tf.random.uniform((N_r,1),-1,1, dtype=DTYPE)*tf.sqrt(1-tf.pow(x_r,2))
X_r = tf.concat([ x_r, y_r, t_r], axis=1)

print("Setting up model")

model = PhysicsInformedNN(lb, ub, layers1, u_data, X_data, X_r)


def time_step():
        loss = model.loss(model.X, model.X0, model.u0)
        return loss

variables = model.model.trainable_variables

cor=75
tol=1.0  * np.finfo(float).eps
iter=70000
fun=100000
ls=100

dde.optimizers.config.set_LBFGS_options(maxcor=cor, ftol=tol,  maxiter=iter, maxfun=fun, maxls=ls)

print("Optimizing model")

start = time.time()

    
dde.optimizers.tfp_optimizer.lbfgs_minimize(variables, time_step)

stop = time.time()

duration = stop-start

print("End trainign, time:",duration)

print("Exporting model")

model.model.export('PINN_Export3')
