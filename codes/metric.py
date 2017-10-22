#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:50:32 2017

@author: PeterTsai
"""

import time
import tensorflow as tf
import numpy as np
import pandas as pd 

def loss_x_entropy(output, target):
    """Cross entropy loss
    
    See https://en.wikipedia.org/wiki/Cross_entropy

    Args:
        output: tensor of net output
        target: tensor of net we are trying to reconstruct
    Returns:
        Scalar tensor of cross entropy
        
"""
    with tf.name_scope("xentropy_loss"):
        net_output_tf = tf.convert_to_tensor(output, name='input')
        target_tf = tf.convert_to_tensor(target, name='target')
        cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'),
                                           target_tf),
                             tf.multiply(tf.log(1 - net_output_tf),
                                    (1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                   name='xentropy_mean')

def multiClassEvaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
        """    
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accu  = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    error = tf.subtract(1.0,accu)
    return accu, error

def calculateAccu(Y_pred,inst_pred,test_Y,test_label,dataset):
          
    
    KP_pred = np.zeros((len(Y_pred),5))
    for bagIdx in range(len(Y_pred)):
        for k in range(len(dataset.k)):
            if Y_pred[bagIdx] in dataset.C5k_CLASS[k]:
                c = dataset.C5k_CLASS[k].index(Y_pred[bagIdx])
                kinst = np.argmax(inst_pred[k][bagIdx,:,c])
                KP_pred[bagIdx] = dataset.playerMap[k][kinst]
                
    Y_correct = np.equal(Y_pred,np.argmax(test_Y,1))
    bagAccu = np.sum(Y_correct) / Y_pred.size
    
    y_correct = np.equal(KP_pred[Y_correct,:],test_label[Y_correct,:])
    
    pAccu = np.sum(y_correct) / KP_pred[Y_correct,:].size
    print('bag accuracy %.5f, inst accuracy %.5f' %(bagAccu, pAccu))
    time.sleep(1)
    return bagAccu, pAccu   
   
        
def ConfusionMatrix(logits,labels,dataset,filename):
    C = np.zeros((len(dataset.tacticName),len(dataset.tacticName)))
    CM = C    
    flattenC5k = [val for sublist in dataset.C5k_CLASS for val in sublist]
    for bagIdx in range(len(labels)):
        gt = np.argmax(labels[bagIdx])
        #pred = np.argmax(logits[bagIdx])
        pred = logits[bagIdx]
        new_gt = flattenC5k[gt]
        new_pred= flattenC5k[pred]
        C[new_gt,new_pred] = C[new_gt,new_pred] + 1
        
    print(C)
    cumC = np.sum(C,axis=1)
    
    for p in range(len(C)):
        CM[p,:] = np.divide(C[p,:],cumC[p])
    
    df = pd.DataFrame(CM)
    df.round(3)
    if filename is not None:
        df.to_csv(filename)