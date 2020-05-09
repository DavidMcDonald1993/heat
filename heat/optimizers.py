import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.python.training import optimizer

def minkowski_dot(x, y):
	axes = len(x.shape) - 1, len(y.shape) -1
	return K.batch_dot(x[...,:-1], y[...,:-1], axes=axes) - \
		K.batch_dot(x[...,-1:], y[...,-1:], axes=axes)

class ReimannianOptimizer(optimizer.Optimizer):
	
    def __init__(self, 
        lr=0.1, 
        use_locking=False,
        name="ReimannianOptimizer"):
        super(ReimannianOptimizer, self).\
            __init__(use_locking, name)
        self.lr = lr

    def _apply_dense(self, grad, var):
        spacial_grad = grad[:,:-1]
        t_grad = -grad[:,-1:]
        
        ambient_grad = tf.concat([spacial_grad, t_grad], 
            axis=-1)
        tangent_grad = self.project_onto_tangent_space(var, 
            ambient_grad)
        
        exp_map = self.exponential_mapping(var, 
            - self.lr * tangent_grad)
        
        return tf.assign(var, exp_map)
        
    def _apply_sparse(self, grad, var):
        indices = grad.indices
        values = grad.values

        p = tf.gather(var, indices, name="gather_apply_sparse")

        spacial_grad = values[:, :-1]
        t_grad = -values[:, -1:]

        ambient_grad = tf.concat([spacial_grad, t_grad], 
            axis=-1, 
            name="optimizer_concat")

        tangent_grad = self.project_onto_tangent_space(p, 
            ambient_grad)
        
        exp_map = self.exponential_mapping(p, 
            - self.lr * tangent_grad)

        return tf.scatter_update(ref=var, 
            indices=indices, updates=exp_map, 
            name="scatter_update")

    def project_onto_tangent_space(self, 
        hyperboloid_point, minkowski_ambient):
        return minkowski_ambient + \
            minkowski_dot(hyperboloid_point, minkowski_ambient) * \
                hyperboloid_point

    def normalise_to_hyperboloid(self, x):
        return x / K.sqrt( K.abs(minkowski_dot(x, x)) )

    def exponential_mapping( self, p, x ):
        r = K.sqrt( K.relu( minkowski_dot(x, x) ) ) 
        ####################################################
        exp_map_p = tf.cosh(r) * p
        
        idx = tf.cast( 
            tf.where(r > K.cast(0., K.floatx()), )[:,0], 
            tf.int64)
        non_zero_norm = tf.gather(r, idx)
        z = tf.gather(x, idx) / non_zero_norm

        updates = tf.sinh(non_zero_norm) * z
        dense_shape = tf.cast( tf.shape(p), tf.int64)
        exp_map_x = tf.scatter_nd(indices=idx[:,None], 
            updates=updates, shape=dense_shape)
        
        exp_map = exp_map_p + exp_map_x 
        #####################################################
        # z = x / K.maximum(norm_x, K.epsilon()) # unit norm 
        # exp_map = tf.cosh(norm_x) * p + tf.sinh(norm_x) * z
        #####################################################
         
        # account for floating point imprecision
        exp_map = self.normalise_to_hyperboloid(exp_map)

        return exp_map