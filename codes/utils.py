#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 23:09:38 2017

@author: PeterTsai
"""
import tensorflow as tf

def get_rnn_type(op):
    return {
            'BasicRNN' : tf.contrib.rnn.BasicRNNCell,
            'BasicLSTM': tf.contrib.rnn.BasicLSTMCell,
            'LSTM'     : tf.contrib.rnn.LSTMCell,
            'GRU'      : tf.contrib.rnn.GRUCell,
            'LayerNormBasicLSTM': tf.contrib.rnn.LayerNormBasicLSTMCell
        }[op]

def get_optimizer(op):
    return {
        'RMSProp': tf.train.RMSPropOptimizer,
        'Adam'   : tf.train.AdamOptimizer,
        'GD'     : tf.train.GradientDescentOptimizer 
        }[op]

def get_activation_fn(op):
    return {
        'tanh' :  tf.nn.tanh,
        'sigmoid': tf.nn.sigmoid,
        'relu6' : tf.nn.relu6,
        'softmax' : tf.nn.softmax,
        'relu'    : tf.nn.relu,
        'log_sigmoid':tf.log_sigmoid,
        'linear' : tf.identity
        }[op]

def showProperties(obj):
    properties = obj.__dict__
    for i in properties:
        print('{0}: {1}'.format(i, properties[i]))       

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
        
def xavier_initializer(shape, uniform=False):
    if shape:
        fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        fan_out = float(shape[-1])
    else:
        fan_in = 1.0
        fan_out = 1.0
        
    n = (fan_in + fan_out) / 2.0                
    if uniform:
        limit = tf.sqrt(3.0 / n)
        return tf.random_uniform(shape, -limit, limit) 
    else:
        # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
        trunc_stddev = tf.sqrt(1.3 / n)
        return tf.truncated_normal(shape, 0.0, trunc_stddev)            

def printLog(log_file, *args, **kwargs):
    print(*args, **kwargs)
    with open(log_file,'a') as file:
        print(*args, **kwargs, file=file)

if __name__ == "__main__":
    clear_all()