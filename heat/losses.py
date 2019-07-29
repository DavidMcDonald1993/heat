import numpy as np
import tensorflow as tf 
import keras.backend as K

def minkowski_dot(x, y):
    axes = len(x.shape) - 1, len(y.shape) -1
    return K.batch_dot(x[...,:-1], y[...,:-1], axes=axes) - K.batch_dot(x[...,-1:], y[...,-1:], axes=axes)

def hyperbolic_sigmoid_loss(y_true, y_pred):

    u_emb = y_pred[:,0]
    samples_emb = y_pred[:,1:]

    inner_uv = minkowski_dot(u_emb, samples_emb) 

    pos_p_uv = tf.nn.sigmoid(inner_uv[:,0])
    neg_p_uv = 1 - tf.nn.sigmoid(inner_uv[:,1:])

    pos_p_uv = K.clip(pos_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())
    neg_p_uv = K.clip(neg_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())

    return - K.mean( K.log( pos_p_uv ) + K.sum( K.log( neg_p_uv ), axis=-1) )

def hyperbolic_softmax_loss(sigma=1.):

    def loss(y_true, y_pred, sigma=sigma):

        source_node_embedding = y_pred[:,0]
        target_nodes_embedding = y_pred[:,1:]
        
        inner_uv = minkowski_dot(source_node_embedding, target_nodes_embedding) 
        inner_uv = -inner_uv
        inner_uv = K.maximum(inner_uv, 1. + K.epsilon())

        d_uv = tf.acosh(inner_uv) 
        minus_d_uv_sq = - 0.5 * K.square(d_uv / sigma)

        return K.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true[:,0,0], logits=minus_d_uv_sq)) 

    return loss

def hyperbolic_hypersphere_loss(r, t):

    def loss(y_true, y_pred, r=r, t=t):

        u_emb = y_pred[:,0]
        samples_emb = y_pred[:,1:]
        
        inner_uv = minkowski_dot(u_emb, samples_emb) 
        inner_uv = -inner_uv - 1. + 1e-7
        inner_uv = K.maximum(inner_uv, K.epsilon()) # clip to avoid nan

        d_uv = tf.acosh(1. + inner_uv) 
        d_uv_sq = K.square(d_uv)

        r_sq = K.square(r)
        # r_sq = K.stop_gradient( K.mean(d_uv_sq) )

        out_uv = (r_sq - d_uv_sq) / t

        pos_out_uv = out_uv[:,0]
        neg_out_uv = out_uv[:,1:]
        
        pos_p_uv = tf.nn.sigmoid(pos_out_uv)
        neg_p_uv = 1 - tf.nn.sigmoid(neg_out_uv)

        pos_p_uv = K.clip(pos_p_uv, min_value=1e-7, max_value=1-1e-7)
        neg_p_uv = K.clip(neg_p_uv, min_value=1e-7, max_value=1-1e-7)

        return - K.mean( K.log( pos_p_uv ) + K.sum( K.log( neg_p_uv ), axis=-1))

    return loss