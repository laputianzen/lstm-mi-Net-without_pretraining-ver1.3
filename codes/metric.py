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

def calulcutePAccuTF(Y,y_playerPool,Y_placeholder,y_placeholder):
    bool_YMask = tf.one_hot(tf.argmax(Y,axis=-1),int(Y.shape[1]),axis=-1)
    bool_YMask = tf.cast(bool_YMask,tf.bool)
    y = tf.boolean_mask(y_playerPool,bool_YMask)        
    correct_prediction = tf.equal(tf.argmax(Y,axis=1), tf.argmax(Y_placeholder,axis=1))
    y_correctY = tf.boolean_mask(y,correct_prediction)
    y_placeholder_correctY = tf.cast(tf.boolean_mask(y_placeholder,correct_prediction),tf.int32)
    y_correct = tf.equal(y_correctY,y_placeholder_correctY)
    NUM_PLAYER = int(y_placeholder.shape[1])
    y_accu = tf.reduce_sum(tf.cast(y_correct,tf.float32))/(tf.reduce_sum(tf.cast(correct_prediction,tf.float32))*NUM_PLAYER)
    return y_accu
      

def calculateAccu(Y_pred,inst_pred,test_Y,test_label,dataset):
    
    num_k = np.zeros(len(dataset.tacticName),dtype=np.int8)
    IdxMap = []
    tmp = []
    k = 0
    boundary = len(dataset.C5k_CLASS[0])
    for idx in range(len(dataset.tacticName)):
        if idx >= boundary:
            k = k + 1            
            boundary = boundary + len(dataset.C5k_CLASS[k])
            IdxMap.append(tmp)
            tmp = []
        num_k[idx] = k
        tmp.append(idx)
    IdxMap.append(tmp)
    
    KP_pred = np.zeros((len(Y_pred),5))
    for bagIdx in range(len(Y_pred)):
        k = num_k[Y_pred[bagIdx]]
        c = IdxMap[k].index(Y_pred[bagIdx])
        kinst = np.argmax(inst_pred[k][:,bagIdx,c])
        KP_pred[bagIdx] = dataset.playerMap[k][kinst]
# =============================================================================
#     for bagIdx in range(len(Y_pred)):
#         for k in range(len(dataset.k)):
#             if Y_pred[bagIdx] in dataset.C5k_CLASS[k]:
#                 c = dataset.C5k_CLASS[k].index(Y_pred[bagIdx])
#                 kinst = np.argmax(inst_pred[k][:,bagIdx,c])
#                 #kinst = np.argmax(inst_pred[k][bagIdx,:,c])
#                 KP_pred[bagIdx] = dataset.playerMap[k][kinst]
# =============================================================================
                
    Y_correct = np.equal(Y_pred,np.argmax(test_Y,1))
    bagAccu = np.sum(Y_correct) / Y_pred.size
    
    y_correct = np.equal(KP_pred[Y_correct,:],test_label[Y_correct,:])
    
    pAccu = np.sum(y_correct) / KP_pred[Y_correct,:].size
    #print(y_correct)
    print('bag accuracy %.5f, inst accuracy %.5f' %(bagAccu, pAccu))
    time.sleep(1)
    return bagAccu, pAccu   
   
        
def ConfusionMatrix(logits,labels,dataset,filename,text_file):
    C = np.zeros((len(dataset.tacticName),len(dataset.tacticName)))
    CM = C    
    #flattenC5k = [val for sublist in dataset.C5k_CLASS for val in sublist]
    for bagIdx in range(len(labels)):
        gt = np.argmax(labels[bagIdx])
        #pred = np.argmax(logits[bagIdx])
        pred = logits[bagIdx]
        C[gt,pred] = C[gt,pred] + 1
        ''' convert tactic from network order to dataset order '''
# =============================================================================
#         new_gt = flattenC5k[gt]
#         new_pred= flattenC5k[pred]
#         C[new_gt,new_pred] = C[new_gt,new_pred] + 1
# =============================================================================
        
    print(C)
    text_file.write(np.array2string(C))
    cumC = np.sum(C,axis=1)
    
    for p in range(len(C)):
        CM[p,:] = np.divide(C[p,:],cumC[p])
    
    df = pd.DataFrame(CM)
    df.round(3)
    if filename is not None:
        df.to_csv(filename)